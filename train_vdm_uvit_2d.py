import torch
import torchinfo
import os
from pathlib import Path

from torch.utils.data import Subset

from monai.bundle import ConfigParser
from monai.utils import set_determinism
from datetime import datetime
from ema_pytorch import EMA

import copy
import argparse
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np


from monai.networks.nets import VQVAE
from dataset_2d import setup_datasets_diffusion, setup_dataloaders
import pdb

import json, matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms as T
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.fid import FIDMetric
from torchvision.models import inception_v3

from models.lvdm.uvit_2d import UViT
from models.lvdm.vdm_2d import VDM
from models.lvdm.utils import (
    DeviceAwareDataLoader,
    TrainConfig,
    check_config_matches_checkpoint,
    cycle,
    evaluate_model_and_log,
    get_date_str,
    handle_results_path,
    has_int_squareroot,
    init_config_from_args,
    init_logger,
    log,
    make_cifar,
    print_model_summary,
    sample_batched,
)


class Trainer:
    def __init__(
        self,
        diffusion_model,
        ae_model_ct,
        ae_model_cbct,
        train_set,
        val_set,
        test_set,
        accelerator,
        optimizer,
        cfg,
        num_steps=100_000,
        ema_decay=0.9999,
        ema_update_every=1,
        ema_power=3/4,
        save_and_eval_every=1000,
        num_samples=8,
        results_path="./results",
        resume=False,
        clip_samples=False,
        num_classes=None
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.ae_model_ct = ae_model_ct
        self.ae_model_cbct = ae_model_cbct
        self.train_set = train_set
        self.val_set = val_set,
        self.test_set = test_set,
        self.accelerator = accelerator
        self.save_and_eval_every = save_and_eval_every
        self.num_samples = num_samples
        self.cfg = cfg
        self.num_steps = num_steps
        self.clip_samples = clip_samples
        self.step = 0
        
        self.opt = optimizer
        self.writer = None
        
        experiment_name = cfg["default"]["experiment_name"]
        
        if cfg["default"]["make_logs"]:
            if "experiment_name" in cfg.default.keys():
                experiment_name = cfg["default"]["experiment_name"]
            else:
                experiment_name = f"diffusion@{datetime.now().strftime('%d.%m.%Y-%H:%M')}"
            self.experiment_dir = Path(cfg["default"]["checkpoint_dir"]) / experiment_name

            self.experiment_dir.mkdir(exist_ok=True, parents=True)
            cfg.export_config_file(cfg.get_parsed_content(), os.path.join(self.experiment_dir, "config.yaml"), fmt="yaml")
            self.writer = SummaryWriter(self.experiment_dir, "tb")


        def make_dataloader(dataset, limit_size=None, *, train=False):
            if limit_size is not None:
                dataset = Subset(dataset, range(limit_size))
            dataloader = DeviceAwareDataLoader(
                dataset,
                cfg["dataset"]["train_batch_size"],
                shuffle=train,
                pin_memory=True,
                num_workers=cfg["dataset"]["num_workers"],
                drop_last=True,
                device=accelerator.device if not train else None,  # None -> standard DL
            )
            if train:
                dataloader = accelerator.prepare(dataloader)
            return dataloader

        self.train_dataloader = cycle(make_dataloader(train_set, train=True))
        #self.train_dataloader = make_dataloader(train_set, train=True)
        self.validation_dataloader = make_dataloader(val_set)
        self.test_dataloader = make_dataloader(test_set, len(val_set))

        self.diffusion_model = accelerator.prepare(diffusion_model)
        self.opt = accelerator.prepare(optimizer)
        
    def train(self):
        # Create a single progress bar for steps
        with tqdm(
            range(self.num_steps),
            desc="Training",
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.num_steps:
                # Get next batch using next()
                data = next(self.train_dataloader)
                cbct_img, ct_img = data["cbct"], data["ct"]
                with torch.no_grad():
                    z = self.ae_model_ct.encode_stage_2_inputs(ct_img)            
                    z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct_img)
                self.opt.zero_grad()
                loss, metrics_tr = self.diffusion_model(z, z_cond, ct_img)
                self.accelerator.backward(loss)

                self.opt.step()
                pbar.set_description(f"loss: {loss.item():.4f}")
                
                self.step += 1
                self.accelerator.wait_for_everyone()
                
                if self.accelerator.is_main_process:
                    if hasattr(self, 'ema'):
                        self.ema.update()      
                    if self.step % self.save_and_eval_every == 0:
                        #pdb.set_trace()
                        self.validation()
                        self.eval()
                        print(f"Completed step {self.step}/{self.num_steps}")
                pbar.update()
                #pdb.set_trace()

                if self.step % 100 == 0:
                    self.writer.add_scalar("train/diff_loss", metrics_tr["diff_loss"].item(), self.step)
                    self.writer.add_scalar("train/latent_loss", metrics_tr["latent_loss"].item(), self.step)
                    self.writer.add_scalar("train/recon_loss", metrics_tr["recon_loss"].item(), self.step)


    def validation(self):
        """Evaluate the model by sampling from the diffusion process and calculating metrics."""
        #pdb.set_trace()
        self.accelerator.unwrap_model(self.diffusion_model)
        self.diffusion_model.eval()

        # Initialize metrics for 2D data
        psnr_metric = PSNRMetric(max_val=1.0)  # For data in range [-1, 1]
        ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=2)  # Changed to 2D
        
        all_metrics = {
            "val_psnr": [],
            "val_ssim": [],
            "val_loss": []
        }
        
        with torch.no_grad():
            # Process validation set in batches
            for data in self.validation_dataloader:
                cbct_img, ct_img = data["cbct"], data["ct"]
                
                z_target = self.ae_model_ct.encode_stage_2_inputs(ct_img)
                z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct_img)
                
                loss, metrics = self.diffusion_model(z_target, z_cond, ct_img)

                
                sampled_z = self.sample_conditional(z_cond, self.cfg["n_sample_steps"])
                
                decoded_sampled = self.ae_model_ct.decode_stage_2_outputs(sampled_z)
                decoded_target = self.ae_model_ct.decode_stage_2_outputs(z_target)
                
                # For 2D data, we don't need to select middle slice
                psnr_metric(decoded_sampled, decoded_target)
                ssim_metric(decoded_sampled, decoded_target)
                
                all_metrics["val_loss"].append(loss.item())
        
        # Compute average metrics
        val_psnr = psnr_metric.aggregate().item()
        val_ssim = ssim_metric.aggregate().item()
        val_loss = sum(all_metrics["val_loss"]) / len(all_metrics["val_loss"])
        
        # Log metrics to tensorboard if available
        if self.writer is not None:
            self.writer.add_scalar("validation/loss", val_loss, self.step)
            self.writer.add_scalar("validation/psnr", val_psnr, self.step)
            self.writer.add_scalar("validation/ssim", val_ssim, self.step)
            
            # Log image samples for 2D data
            if decoded_sampled.shape[0] > 0:
                # Create grid of sample images: condition, generated, ground truth
                img_grid = make_grid([
                    cbct_img[0],  # condition
                    decoded_sampled[0],  # generated
                    decoded_target[0]  # ground truth
                ], nrow=3, normalize=True)
                self.writer.add_image("validation/samples", img_grid, self.step)
        
        # Print metrics
        print(f"\nValidation @ step {self.step}: PSNR={val_psnr:.4f}, SSIM={val_ssim:.4f}, Loss={val_loss:.4f}")
        
        # Save checkpoint if needed
        if self.cfg["default"]["make_logs"]:
            self.save_checkpoint()
        
        # Return model to training mode
        self.diffusion_model.train()
        
        return val_loss

    
    def eval(self):
        """
        Evaluate the diffusion model on the held‑out test_set.

        Outputs
        -------
        <exp_dir>/test_metrics.json   # psnr, ssim, mse, fid
        <exp_dir>/qualitative.png     # CBCT | generated‑CT | GT‑CT | |diff|
        """


        self.accelerator.unwrap_model(self.diffusion_model)
        self.diffusion_model.eval()

        # ─── metric objects ────────────────────────────────────────────────────────
        psnr = PSNRMetric(max_val=1.0)
        ssim = SSIMMetric(data_range=1.0, spatial_dims=2)
        fid_metric = FIDMetric()

        mse_running, n_seen = 0.0, 0

        # Inception‑v3 feature extractor for FID (pool‑3 layer, 2048‑D)
        
        inception = inception_v3(pretrained=True, transform_input=False).to(
            self.accelerator.device
        )
        inception.fc = torch.nn.Identity()
        inception.eval()

        def inception_feats(x: torch.Tensor) -> torch.Tensor:
            # x: (B,1,H,W) in [-1,1]  ➜ (B,2048)
            x = (x.clamp(-1, 1) + 1) / 2
            x = x.repeat(1, 3, 1, 1)                                      # 1‑ch → 3‑ch
            with torch.no_grad():
                z = inception(x)
            return z.flatten(1)                                           # (B,2048)

        feats_real, feats_fake = [], []
        samples_done = 0
        saved_grid   = False
        qual_imgs = []  

        # ─── loop through the test set ────────────────────────────────────────────
        for batch in self.test_dataloader:
            cbct, ct_gt = batch["cbct"], batch["ct"]

            with torch.no_grad():
                z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct)
                ct_gen = self.ae_model_ct.decode_stage_2_outputs(
                    self.sample_conditional(z_cond, self.cfg["n_sample_steps"])
                )

            # metrics …
            psnr(ct_gen, ct_gt)
            ssim(ct_gen, ct_gt)
            mse_running += F.mse_loss(ct_gen, ct_gt, reduction="sum").item()
            n_seen      += ct_gt.numel()

            feats_fake.append(inception_feats(ct_gen))
            feats_real.append(inception_feats(ct_gt))

            # collect qualitative slices (move to CPU so GPU can be freed)
            for i in range(cbct.size(0)):
                if len(qual_imgs) >= 4 * self.num_samples:
                    break
                diff = (ct_gen[i] - ct_gt[i]).abs()
                for tensor in (cbct[i], ct_gen[i], ct_gt[i], diff):
                    qual_imgs.append(tensor.detach().cpu())      # <-- off‑load now

            samples_done += cbct.size(0)
            if samples_done >= self.num_samples:
                break
        
        if qual_imgs:
            grid = make_grid(
                torch.stack(qual_imgs, 0),     # shape (4·N, 1, H, W)
                nrow=4,
                normalize=False,
                value_range=(0, 1)            # keep CT intensities consistent
            )
            plt.figure(figsize=(10, 3 * self.num_samples))
            plt.axis("off")
            plt.imshow(grid.permute(1, 2, 0), cmap="gray")
            plt.savefig(self.experiment_dir / f"qualitative_step-{self.step}.png",
                        bbox_inches="tight", dpi=200)
            plt.close()

        # ─── reduce & log ─────────────────────────────────────────────────────────
        psnr_val = psnr.aggregate().item()
        ssim_val = ssim.aggregate().item()
        mse_val  = mse_running / n_seen
        fid_val  = fid_metric(torch.vstack(feats_fake),
                            torch.vstack(feats_real)).item()

        results = dict(psnr=psnr_val, ssim=ssim_val, mse=mse_val, fid=fid_val)
        (self.experiment_dir / f"test_metrics_{self.step}.json").write_text(
            json.dumps(results, indent=2)
        )

        print(f"\nTEST ▸  PSNR {psnr_val:.3f}  SSIM {ssim_val:.3f}  "
            f"MSE {mse_val:.5f}  FID {fid_val:.3f}")

        self.diffusion_model.train()

    def sample_conditional(self, z_cond, n_sample_steps):
        """Sample from the diffusion model using conditioning with a progress bar."""
        # Get batch size from conditioning
        batch_size = z_cond.shape[0]
        
        z = torch.randn((batch_size, *self.diffusion_model.image_shape), device=self.accelerator.device)
        
        steps = torch.linspace(1.0, 0.0, n_sample_steps + 1, device=self.accelerator.device)
        
        with torch.no_grad():

            disable_pbar = not self.accelerator.is_main_process
            
            for i in tqdm(range(n_sample_steps), desc="Sampling", leave=False, disable=disable_pbar):

                z = self.diffusion_model.sample_p_s_t(
                    z, 
                    steps[i], 
                    steps[i + 1], 
                    clip_samples=self.clip_samples, 
                    context=z_cond
                )
        
        return z

    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step,
            "model": self.accelerator.unwrap_model(self.diffusion_model).state_dict(),
            "opt": self.opt.state_dict(),
        }
        
        # Add EMA model if present
        if hasattr(self, 'ema') and self.accelerator.is_main_process:
            checkpoint["ema"] = self.ema.state_dict()
        
        torch.save(checkpoint, self.experiment_dir / f"model_{self.step}.pt")
        torch.save(checkpoint, self.experiment_dir / "latest.pt")
        
        # Save configuration
        config_path = self.experiment_dir / "config.yaml"
        if not config_path.exists():
            self.cfg.exconfig.export_config_file(
                self.cfg.get_parsed_content(), 
                str(config_path), 
                fmt="yaml"
            )
        

def main(cfg):
    
    # Override config with command line arguments if provided
    DEVICE = cfg["default"]["device"]
    device = torch.device(DEVICE)
    seed = cfg["default"]["device"]
    set_determinism(seed=seed)

    # Check if CBCT VQVAE config and checkpoint files exist
    cbct_config_path = cfg["paths"]["cbct_vqvae_config"]
    cbct_checkpoint_path = cfg["paths"]["cbct_vq_checkpoint"]
    
    if not os.path.exists(cbct_config_path):
        raise FileNotFoundError(f"CBCT VQVAE config file not found: {cbct_config_path}")
    if not os.path.exists(cbct_checkpoint_path):
        raise FileNotFoundError(f"CBCT VQVAE checkpoint file not found: {cbct_checkpoint_path}")
    
    # Check if CT VQVAE config and checkpoint files exist
    ct_config_path = cfg["paths"]["ct_vqvae_config"]
    ct_checkpoint_path = cfg["paths"]["ct_vq_checkpoint"]
    
    if not os.path.exists(ct_config_path):
        raise FileNotFoundError(f"CT VQVAE config file not found: {ct_config_path}")
    if not os.path.exists(ct_checkpoint_path):
        raise FileNotFoundError(f"CT VQVAE checkpoint file not found: {ct_checkpoint_path}")

    print(f"Loading CBCT VQVAE from:")
    print(f"  Config: {cbct_config_path}")
    print(f"  Checkpoint: {cbct_checkpoint_path}")
    
    print(f"Loading CT VQVAE from:")
    print(f"  Config: {ct_config_path}")
    print(f"  Checkpoint: {ct_checkpoint_path}")

    # Load CBCT VQVAE configuration
    cbct_vqvae_config = ConfigParser()
    cbct_vqvae_config.read_config(cbct_config_path)

    # Load CT VQVAE configuration  
    ct_vqvae_config = ConfigParser()
    ct_vqvae_config.read_config(ct_config_path)

    # Create CBCT autoencoder using CBCT config
    cbct_num_channels_tuple = tuple(
        int(x) for x in cbct_vqvae_config["vqvae"]["num_channels"].split(', ')
    )
    cbct_downsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in cbct_vqvae_config["vqvae"]["downsample_parameters"].values()
    )
    cbct_upsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in cbct_vqvae_config["vqvae"]["upsample_parameters"].values()
    )

    cbct_ae = VQVAE(
        spatial_dims=int(cbct_vqvae_config["vqvae"]["spatial_dims"]),
        in_channels=int(cbct_vqvae_config["vqvae"]["in_channels"]),
        out_channels=int(cbct_vqvae_config["vqvae"]["out_channels"]),
        channels=cbct_num_channels_tuple,
        num_res_channels=int(cbct_vqvae_config["vqvae"]["num_res_channels"]),
        num_res_layers=int(cbct_vqvae_config["vqvae"]["num_res_layers"]),
        downsample_parameters=cbct_downsample_tuple,
        upsample_parameters=cbct_upsample_tuple,
        num_embeddings=int(cbct_vqvae_config["vqvae"]["num_embeddings"]),  # codebook length
        embedding_dim=int(cbct_vqvae_config["vqvae"]["embedding_dim"])
    )

    # Create CT autoencoder using CT config
    ct_num_channels_tuple = tuple(
        int(x) for x in ct_vqvae_config["vqvae"]["num_channels"].split(', ')
    )
    ct_downsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in ct_vqvae_config["vqvae"]["downsample_parameters"].values()
    )
    ct_upsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in ct_vqvae_config["vqvae"]["upsample_parameters"].values()
    )

    ct_ae = VQVAE(
        spatial_dims=int(ct_vqvae_config["vqvae"]["spatial_dims"]),
        in_channels=int(ct_vqvae_config["vqvae"]["in_channels"]),
        out_channels=int(ct_vqvae_config["vqvae"]["out_channels"]),
        channels=ct_num_channels_tuple,
        num_res_channels=int(ct_vqvae_config["vqvae"]["num_res_channels"]),
        num_res_layers=int(ct_vqvae_config["vqvae"]["num_res_layers"]),
        downsample_parameters=ct_downsample_tuple,
        upsample_parameters=ct_upsample_tuple,
        num_embeddings=int(ct_vqvae_config["vqvae"]["num_embeddings"]),  # codebook length
        embedding_dim=int(ct_vqvae_config["vqvae"]["embedding_dim"])
    )
    cbct_ae.load_state_dict(torch.load(cfg["paths"]["cbct_vq_checkpoint"])["model_state_dict"])
    ct_ae.load_state_dict(torch.load(cfg["paths"]["ct_vq_checkpoint"])["model_state_dict"])

    cbct_ae.to(device)
    ct_ae.to(device)

    cbct_ae.eval()
    ct_ae.eval()

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)

    # Use CT VQGAN directory for dataset indices
    stage_1_idxs_file = Path(cfg["paths"]["ct_vq_checkpoint"]).parent / "dataset_indices.json"
    
    # Check if dataset indices file exists
    if not stage_1_idxs_file.exists():
        raise FileNotFoundError(f"Dataset indices file not found: {stage_1_idxs_file}")
    train_dataset, val_dataset, test_dataset = setup_datasets_diffusion(cfg, stage_1_idxs_file)

    train_loader, _ = setup_dataloaders(cfg, save_train_idxs=False)

    check_data = next(iter(train_loader))

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            z = ct_ae.encode_stage_2_inputs(check_data["ct"].to(device))
            print(z.shape)
    del train_loader

    print(f"Codebook latent shape: {z.shape}")
    model = UViT(
        img_size=cfg["uvit"]["img_size"],
        patch_size=cfg["uvit"]["patch_size"],
        in_chans=cfg["uvit"]["in_chans"],
        embed_dim=cfg["uvit"]["embed_dim"],
        depth=cfg["uvit"]["depth"],
        num_heads=cfg["uvit"]["num_heads"],
        conv=cfg["uvit"],
        gamma_max=cfg["gamma_max"],
        gamma_min=cfg["gamma_min"],
    )
    
    diffusion = VDM(
        model,
        cfg,
        ct_ae,
        image_shape=z[0].shape
    )
    
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        eval(cfg["optim"]["lr"]),
        betas=(0.9, 0.99),
        weight_decay=cfg["optim"]["weight_decay"],
        eps=1e-8
    )

    Trainer(
        diffusion,
        ct_ae,
        cbct_ae,
        train_dataset,
        val_dataset,
        test_dataset,
        accelerator,
        optimizer,
        cfg,
        num_steps=cfg["num_steps"],
        save_and_eval_every=cfg["save_and_eval_every"],
    ).train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start train")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to configuration *.yaml file",
        required=False,
        default="configs/diffusion_config.yaml"
    )

    args = parser.parse_args()

    config = ConfigParser()
    config.read_config(args.config)

    main(config)
