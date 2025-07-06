import pathlib
import torch
from typing import Dict, List, Tuple, Any

import numpy as np
from monai.data import DataLoader, Dataset
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    ToTensord,
    Resized,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    ScaleIntensityd,
    MapTransform,
    RandAdjustContrastd,
    RandAffined
)

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pdb
import os
import json
import json


import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
import multiprocessing




def load_dataset_indices(indices_path: str) -> Dict[str, List[int]]:
    """
    Load dataset split indices from a JSON file.
    
    Args:
        indices_path: Path to the JSON file containing dataset indices
        
    Returns:
        Dictionary containing train, validation and test indices
    """
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    return indices


def df_to_dict_list(df):
    dict_list = []
    for _, row in df.iterrows():
        # Start with required fields
        dict_entry = {
            "subj_id": row["subj_id"],
            "anatomy": row["anatomy"],
            "mask": pathlib.Path(row["mask"])
        }
        # Add modality paths only if they exist in the dataframe
        if "cbct" in row:
            dict_entry["cbct"] = pathlib.Path(row["cbct"])
        if "ct" in row:
            dict_entry["ct"] = pathlib.Path(row["ct"])
        dict_list.append(dict_entry)
    return dict_list


class ApplyMaskd(MapTransform):
    """
    Apply a mask to the input images.
    
    Args:
        keys: Keys to be used for the input images
        mask_key: Key to be used for the mask
    
    Returns:
        Dict
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str = "mask",
        #fill_value: float = -1000.0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        #self.fill_value = fill_value

    def __call__(self, data):
        d = dict(data)
        mask = d[self.mask_key]  # assumed to be 0/1 float or byte
        inv_mask = 1.0 - mask
        for key in self.keys:
            if key in d:
                img = d[key]
                min_img = img.min()
                # background → fill_value, foreground → original
                d[key] = img * mask + min_img * inv_mask
        return d


class ClipHUValues(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = torch.clamp(d[key], -800, 1500)
        return d


def setup_transforms(config: Dict):
    interpolate = config["dataset"]["interpolate"]
    modality = config["dataset"].get("modality", ["ct", "cbct"])

    # Determine which keys to use based on modality
    image_keys = []
    if "cbct" in modality:
        image_keys.append("cbct")
    if "ct" in modality:
        image_keys.append("ct")
    
    # Always include mask
    all_keys = image_keys + ["mask"]

    if interpolate:
        spatial_size_conf = tuple(config["dataset"]["interpolation_size"])

    val_transf = Compose(
            [
                LoadImaged(
                    keys=all_keys,
                    image_only=True,
                    ensure_channel_first=True,
                    reader="ITKReader"
                ),

                ApplyMaskd(keys=image_keys, mask_key="mask"),
                ClipHUValues(keys=image_keys),
                ScaleIntensityd(
                    keys=image_keys,
                    minv=config["dataset"]["minv"],
                    maxv=config["dataset"]["maxv"]
                ),
                
            ] + ([Resized(keys=all_keys, spatial_size=spatial_size_conf)] if interpolate else []) + \
            [
                EnsureTyped(keys=all_keys, dtype=torch.float),
                ToTensord(keys=all_keys)
            ]
        )

    if config["dataset"]["augment"]:
        try:
            rand_adj_prob = config["dataset"]["rand_adj_contrast"]["prob"]
            rand_adj_gamma = config["dataset"]["rand_adj_contrast"]["gamma"]

            rand_affine = config["dataset"]["rand_affine"]["prob"]
        except:
            pass


        transforms = [
            LoadImaged(
                keys=all_keys,
                image_only=True,
                ensure_channel_first=True,
                reader="ITKReader"
            ),

            ApplyMaskd(keys=image_keys, mask_key="mask"),
            ClipHUValues(keys=image_keys),
            ScaleIntensityd(keys=image_keys, minv=config["dataset"]["minv"], maxv=config["dataset"]["maxv"]),
            
        ] + ([Resized(keys=all_keys, spatial_size=spatial_size_conf)] if interpolate else [])

        if rand_adj_prob > 0.0:
            transforms.append(
                RandAdjustContrastd(
                    keys=image_keys,
                    prob=rand_adj_prob,
                    gamma=tuple(rand_adj_gamma)
                )
            )

        if rand_affine > 0.0:
            transforms.append(
                RandAffined(
                    keys=all_keys,
                    rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                    translate_range=[(-1, 1), (-1, 1)],
                    scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                    padding_mode="zeros",
                    prob=rand_affine
                )
            )

        transforms.extend(
            [
                EnsureTyped(keys=all_keys, dtype=torch.float),
                ToTensord(keys=all_keys)
            ]
        )

        train_transf = Compose(transforms)
    else:
        train_transf = val_transf

    return train_transf, val_transf


def create_datafiles(config: Dict, anatomy: List[str]=["AB", "HN", "TH"], modality: List[str]=["cbct", "ct"]) -> Tuple[List[Dict]]:
    """
    Create a list of dictionaries containing paths to image files based on specified modality and anatomy regions
    
    Args:
        config: Configuration dictionary containing dataset parameters
        anatomy: List of anatomical anatomies to include (e.g., ["ab", "hn", "th"] (abdomen, head-neck, thorax))
        modality: List of modalities to include - can contain "cbct", "ct", or both
    
    Returns:
        List of dictionaries containing file paths and metadata
    """
    #pdb.set_trace()
    files = []
    data_path = pathlib.Path(config["dataset"]["data_path"])

    for mod in modality:
        if mod not in ["cbct", "ct"]:
            raise ValueError("modality must be one of: 'cbct', or 'ct'")

    anatomy = [anat.upper() for anat in anatomy]
    
    for anat in anatomy:
        if anat not in ["AB", "HN", "TH"]:
            raise ValueError(f"anatomy '{anat}' must be one of: 'AB', 'HN', 'TH'")

    for anat in anatomy:
        anat_path = data_path / anat
        if not anat_path.exists():
            print(f"Warning: {anat} directory not found in {data_path}")
            continue

        for subj_dir in anat_path.glob('*'):
            if not subj_dir.is_dir():
                continue

            required_files = {"mask": subj_dir / "mask.mha"}
    
            if "cbct" in modality and "ct" in modality:
                required_files.update({
                    "cbct": subj_dir / "cbct.mha",
                    "ct": subj_dir / "ct.mha"
                })
            elif "cbct" in modality:
                required_files["cbct"] = subj_dir / "cbct.mha"
            elif "ct" in modality:
                required_files["ct"] = subj_dir / "ct.mha"

            if all(p.exists() for p in required_files.values()):
                subj_dict = {
                    "subj_id": str(subj_dir.name),
                    "anatomy": anat,
                }
                subj_dict.update({k: str(v) for k,v in required_files.items()})
                files.append(subj_dict)
    return files



def setup_datasets_diffusion(config: Dict, stage_1_idxs_file) -> DataLoader:
    """
    Create a dataloader for inference using test indices.
    
    Args:
        config: Configuration dictionary containing dataset parameters
        stage_1_idxs_file: json containing 
        
    Returns:
        DataLoader for test dataset
    """
    files = create_datafiles(
        config,
        anatomy=config["dataset"]["anatomy"],
        modality=config["dataset"]["modality"]
    )
    df = pd.DataFrame(files)

    with stage_1_idxs_file.open("r") as f:
        idxs = json.load(f)

    split_dfs = {
        "train": df[df["subj_id"].isin(idxs["train"])].reset_index(drop=True),
        "val":   df[df["subj_id"].isin(idxs["validation"])].reset_index(drop=True),
        "test":  df[df["subj_id"].isin(idxs["test"])].reset_index(drop=True),
    }
    
    # Use validation transforms for inference
    train_transf, val_transf = setup_transforms(config)
    
    train_ds = Dataset(
        data=df_to_dict_list(split_dfs["train"]),
        transform=train_transf
    )

    val_ds = Dataset(
        data=df_to_dict_list(split_dfs["val"]),
        transform=val_transf
    )

    test_ds = Dataset(
        data=df_to_dict_list(split_dfs["test"]),
        transform=val_transf
    )
    
    
    return train_ds, val_ds, test_ds


def setup_dataloaders(config: Dict, save_train_idxs=False):

    files = create_datafiles(
        config,
        anatomy=config["dataset"]["anatomy"],
        modality=config["dataset"]["modality"]
    )
    df = pd.DataFrame(files)

    train_df, temp_df = train_test_split(df, test_size=.3, stratify=df["anatomy"], random_state=config["default"]["random_seed"])

    val_df, hold_out_df = train_test_split(temp_df, test_size=.60, stratify=temp_df["anatomy"], random_state=config["default"]["random_seed"])

    train_transf, val_transf = setup_transforms(config)
    #pdb.set_trace()v
    log_df = pd.concat([train_df, val_df, hold_out_df])

    train_ds = Dataset(
        data=df_to_dict_list(train_df),
        transform=train_transf
    )
    val_ds = Dataset(
        data=df_to_dict_list(val_df),
        transform=val_transf
    )

    test_ds = Dataset(
        data=df_to_dict_list(hold_out_df),
        transform=val_transf
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["dataset"]["train_batch_size"],
        shuffle=config["dataset"]["train_shuffle"],
        num_workers=config["dataset"]["num_workers"],
        #persistent_workers=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config["dataset"]["val_batch_size"],
        shuffle=config["dataset"]["val_shuffle"],
        num_workers=config["dataset"]["num_workers"],
        #persistent_workers=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["dataset"]["val_batch_size"],
        shuffle=config["dataset"]["val_shuffle"],
        num_workers=config["dataset"]["num_workers"],
        #persistent_workers=True
    )

    if save_train_idxs:
        #"""
        # Save indices to experiment folder
        indices = {
            'train': train_df["subj_id"].tolist(),
            'validation': val_df["subj_id"].tolist(),
            'test': hold_out_df["subj_id"].tolist()
        }
        #pdb.set_trace()
        # Save indices if experiment_dir is provided in config
        exp_dir = os.path.join(config["optim"]["checkpoint_dir"], config['default']['experiment_name'])
        indices_path = os.path.join(exp_dir, 'dataset_indices.json')
        with open(indices_path, 'w') as f:
            json.dump(indices, f)
        #pdb.set_trace()"
        #"""
    return train_loader, val_loader


def setup_datasets(config: Dict):
    #pdb.set_trace()

    files = create_datafiles(
        config,
        anatomy=config["dataset"]["anatomy"],
        modality=config["dataset"]["modality"]
    )
    df = pd.DataFrame(files)
    
    #pdb.set_trace()
    train_df, temp_df = train_test_split(df, test_size=.3, stratify=df["anatomy"], random_state=config["default"]["random_seed"])

    val_df, hold_out_df = train_test_split(temp_df, test_size=.60, stratify=temp_df["anatomy"], random_state=config["default"]["random_seed"])

    train_transf, val_transf = setup_transforms(config)
    #pdb.set_trace()v
    log_df = pd.concat([train_df, val_df, hold_out_df])

    train_ds = Dataset(
        data=df_to_dict_list(train_df),
        transform=train_transf
    )
    val_ds = Dataset(
        data=df_to_dict_list(val_df),
        transform=val_transf
    )

    test_ds = Dataset(
        data=df_to_dict_list(hold_out_df),
        transform=val_transf
    )
    
    """
    # Save indices to experiment folder
    indices = {
        'train': train_df["subj_id"].tolist(),
        'validation': val_df["subj_id"].tolist(),
        'test': hold_out_df["subj_id"].tolist()
    }
    #pdb.set_trace()
    # Save indices if experiment_dir is provided in config
    exp_dir = os.path.join("checkpoints", config['default']['experiment_name'])
    indices_path = os.path.join(exp_dir, 'dataset_indices.json')
    with open(indices_path, 'w') as f:
        json.dump(indices, f)
    #pdb.set_trace()"
    """
    return train_ds, val_ds
