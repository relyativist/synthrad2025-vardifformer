import torch
import torch.nn.functional as F
from torch.nn import L1Loss
import argparse
import time
import os
from tqdm.auto import tqdm
from monai.bundle import ConfigParser
from monai.utils import set_determinism

from dataset_2d import setup_dataloaders
from models.lvdm.utils import setup_scheduler, get_lr

from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.networks.nets import VQVAE

from monai.losses.adversarial_loss import PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import Mean
import torchinfo
import pathlib
# pip install monai[nibabel]
# pip install monai[scikit-image]
import pdb
import random


def main(config):
    experiment_name = config["default"].get("experiment_name", "default_experiment")

    checkpoint_dir = pathlib.Path(config["optim"]["checkpoint_dir"]) / experiment_name
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        config.export_config_file(
            config.get_parsed_content(), 
            str(config_path), 
            fmt="yaml"
        )
    
    # Set up tensorboard writer with experiment-specific log directory
    log_dir = checkpoint_dir / 'tb'
    writer = SummaryWriter(log_dir=str(log_dir))

    DEVICE = config["default"]["device"] if isinstance(config["default"].get("device"), int) else "cpu"
    seed = config["default"]["random_seed"] if isinstance(config["default"].get("random_seed"), int) else 42
    set_determinism(seed=seed)

    if DEVICE != 'cpu':
        torch.cuda.set_device(DEVICE)
        print('Using GPU#:', torch.cuda.current_device(), 'Name:', torch.cuda.get_device_name(torch.cuda.current_device()))

    device = torch.device(DEVICE)

    train_loader, val_loader = setup_dataloaders(config, save_train_idxs=True)

    num_channels_tuple = tuple(
        int(x) for x in config["vqvae"]["num_channels"].split(', ')
    )

    downsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in config["vqvae"]["downsample_parameters"].values()
    )
    upsample_tuple =  tuple(
        tuple(
            tuple(v)
        ) for v in config["vqvae"]["upsample_parameters"].values()
    )


    model = VQVAE(
        spatial_dims=int(config["vqvae"]["spatial_dims"]),
        in_channels=int(config["vqvae"]["in_channels"]),
        out_channels=int(config["vqvae"]["out_channels"]),
        channels=num_channels_tuple,
        num_res_channels=int(config["vqvae"]["num_res_channels"]),
        num_res_layers=int(config["vqvae"]["num_res_layers"]),
        downsample_parameters=downsample_tuple,
        upsample_parameters=upsample_tuple,
        num_embeddings=int(config["vqvae"]["num_embeddings"]),  # codebook length
        embedding_dim=int(config["vqvae"]["embedding_dim"]),  # 
        commitment_cost=0.4
    )

    model.to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=int(config["discriminator"]["spatial_dims"]),
        in_channels=int(config["discriminator"]["in_channels"]),
        num_layers_d=int(config["discriminator"]["num_layers_d"]),
        channels=int(config["discriminator"]["num_channels"])
    )

    discriminator.to(device)

    perceptual_loss = PerceptualLoss(
        spatial_dims=int(config["discriminator"]["spatial_dims"]),
        network_type="squeeze")
    perceptual_loss.to(device)

    # Optimizer with lower learning rate
    optimizer_g = torch.optim.Adam(params=model.parameters(), lr=eval(config["optim"]["lr_generator"]))
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=eval(config["optim"]["lr_discriminator"]))
    
    # Cosine learning rate scheduler

    scheduler_g = setup_scheduler(
        optimizer_g,
        config["optim"]["num_epochs"],
        warmup_epochs=0,
        min_lr=eval(config["optim"]["lr_generator"]) * 0.1
    )

    scheduler_d = setup_scheduler(
        optimizer_d,
        config["optim"]["num_epochs"],
        warmup_epochs=0,
        min_lr=eval(config["optim"]["lr_discriminator"]) * 0.1
    )

    start_epoch = 1
    n_epochs = config["optim"]["num_epochs"]
    val_every = config["optim"]["val_every"]

    epbar = tqdm(range(start_epoch, n_epochs + 1), desc='Training Progress', position=0)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    perceptual_weight = 0.001
    
    for epoch in epbar:
        model.train()
        discriminator.train()

        train_metrics = {
                "recon_loss": Mean(device=device),
                "vq_loss": Mean(device=device),
                "gen_loss": Mean(device=device),
                "disc_loss": Mean(device=device),
                "perceptual_loss": Mean(device=device),
                "perplexity": Mean(device=device),
            }

        bbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1)
        for batch_idx, batch_data in enumerate(bbar):
            #pdb.set_trace()
            if len(config["dataset"]["modality"]) > 1:
                modality = random.choice(config["dataset"]["modality"])
                data = batch_data[modality].to(device)
            else:
                modality = config["dataset"]["modality"][0]
                data = batch_data[modality].to(device)
            #pdb.set_trace()
            optimizer_g.zero_grad()
            with torch.amp.autocast("cuda", enabled=config["optim"]["amp"], dtype=torch.bfloat16):
                recon, vq_loss = model(images=data)
                logits_fake = discriminator(recon.contiguous().float())[-1]
                
                recon_loss = l1_loss(recon.float(), data.float())
                p_loss = perceptual_loss(recon.float(), data.float())
                gen_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                loss_g = recon_loss + vq_loss + perceptual_weight * p_loss + adv_weight * gen_loss

            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            with torch.amp.autocast("cuda", enabled=config["optim"]["amp"], dtype=torch.bfloat16):
                logits_fake = discriminator(recon.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)

                logits_real = discriminator(data.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = adv_weight * (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()
            optimizer_d.step()

            # Log losses
            train_metrics["recon_loss"].update(recon_loss)
            train_metrics["vq_loss"].update(vq_loss)
            train_metrics["gen_loss"].update(gen_loss)
            train_metrics["disc_loss"].update(loss_d)
            train_metrics["perceptual_loss"].update(p_loss)
            train_metrics["perplexity"].update(model.quantizer.perplexity)

            bbar.set_postfix(
                {
                    'recon': f'{recon_loss.item():.4f}',
                    'vq': f'{vq_loss.item():.4f}',
                    'gen': f'{gen_loss.item():.4f}',
                    'disc': f'{loss_d.item():.4f}',
                    'modality': modality
                }
            )
        
        train_metric_values = {key: metric.compute().item() for key, metric in train_metrics.items()}

        writer.add_scalar('Train/recon_loss', train_metric_values["recon_loss"], epoch)
        writer.add_scalar('Train/vq_loss', train_metric_values["vq_loss"], epoch)
        writer.add_scalar('Train/gen_loss', train_metric_values["gen_loss"], epoch)
        writer.add_scalar('Train/disc_loss', train_metric_values["disc_loss"], epoch)
        writer.add_scalar('Train/perceptual_loss', train_metric_values["perceptual_loss"], epoch)
        writer.add_scalar('Train/perplexity', train_metric_values["perplexity"], epoch)

        if epoch % val_every == 0:
            model.eval()
            discriminator.eval()

            val_metrics = {
                "recon_loss": Mean(device=device),
                "gen_loss": Mean(device=device),
                "perceptual_loss": Mean(device=device),
            }
            
            with torch.no_grad():
                for val_step, batch_data in enumerate(tqdm(val_loader, desc="Validating", leave=False)):

                    if len(config["dataset"]["modality"]) > 1:
                        modality = random.choice(config["dataset"]["modality"])
                        data = batch_data[modality].to(device)
                    else:
                        modality = config["dataset"]["modality"][0]
                        data = batch_data[modality].to(device)

                    with torch.amp.autocast("cuda", enabled=config["optim"]["amp"], dtype=torch.bfloat16):
                        recon, _ = model(images=data)
                        recon_loss = l1_loss(recon.float(), data.float())
                        p_loss = perceptual_loss(recon.float(), data.float())

                        logits_fake = discriminator(recon.contiguous().float())[-1]
                        gen_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    
                    val_metrics["recon_loss"].update(recon_loss)
                    val_metrics["gen_loss"].update(gen_loss)
                    val_metrics["perceptual_loss"].update(p_loss)
                    
                    if val_step == 0:
                        B, C, H, W = data.shape

                        data_axial = data[...].float()
                        recon_axial = recon[...].float()

                        writer.add_images("val/data_axial", data_axial, global_step=epoch, dataformats='NCHW', )
                        writer.add_images("val/recon_axial", recon_axial, global_step=epoch, dataformats='NCHW')
                   
            val_metric_values = {key: metric.compute().item() for key, metric in val_metrics.items()}
            
            writer.add_scalar('Val/recon_loss', val_metric_values["recon_loss"], epoch)
            writer.add_scalar('Val/gen_loss', val_metric_values["gen_loss"], epoch)
            writer.add_scalar('Val/perceptual_loss', val_metric_values["perceptual_loss"], epoch)

        epbar.set_postfix({
                'recon': f'{train_metric_values["recon_loss"]:.4f}',
                'vq': f'{train_metric_values["vq_loss"]:.4f}',
                'gen': f'{train_metric_values["gen_loss"]:.4f}',
                'disc': f'{train_metric_values["disc_loss"]:.4f}',
                'lr': f'{get_lr(optimizer_g):.2e}',
                'modality': f'{modality}'
            })

        scheduler_g.step()
        scheduler_d.step()

        if epoch % config["optim"]["save_interval"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'scheduler_d_state_dict': scheduler_d.state_dict()
            }, f'{checkpoint_dir}/model_{epoch}.pt')

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start train")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to configuration *.yaml file",
        required=False,
        default="configs/ae_config.yaml"
    )

    args = parser.parse_args()

    config = ConfigParser()
    config.read_config(args.config)

    main(config)
