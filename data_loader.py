import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int) -> None:
    """Ensure deterministic augmentations across dataloader workers."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def check_dataset_structure(data_dir: str) -> List[str]:
    """Check if dataset directory and required files exist."""
    if not os.path.exists(data_dir):
        logger.error(f"Dataset directory '{data_dir}' not found.")
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")
    
    # List all folders in the dataset directory
    patients = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Filter out non-patient folders (e.g., .DS_Store, .venv, etc.)
    patients = [p for p in patients if not p.startswith(('.', '__')) and p not in ('venv', 'data')]
    
    if not patients:
        logger.error(f"No patient folders found in '{data_dir}'.")
        raise ValueError(f"No patient folders found in '{data_dir}'.")
    
    valid_patients: List[str] = []

    for patient in patients:
        img_path = os.path.join(data_dir, patient, f"{patient}_t2.nii.gz")
        mask_path = os.path.join(data_dir, patient, f"{patient}_seg.nii.gz")
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            logger.warning(f"Missing files for {patient}: {img_path} or {mask_path}")
        else:
            valid_patients.append(patient)
    
    logger.info(f"Dataset structure verified: {len(valid_patients)} samples found.")
    return valid_patients

class MRIDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        patients: List[str],
        augment: bool = False,
        augment_params: Optional[Dict[str, float]] = None,
    ):
        self.data_dir = data_dir
        self.patients = patients
        self.augment = augment
        self.augment_params = augment_params or {}

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        case_id = self.patients[idx]
        img_path = os.path.join(self.data_dir, case_id, f"{case_id}_t2.nii.gz")
        mask_path = os.path.join(self.data_dir, case_id, f"{case_id}_seg.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing files for {case_id}: {img_path} or {mask_path}")

        # Load NIfTI files
        img = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize & Resize
        img = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img
        img = resize(img, (32, 32, 32), order=1, preserve_range=True, anti_aliasing=True)
        mask = resize(mask, (32, 32, 32), order=0, preserve_range=True, anti_aliasing=False)

        # Binarize mask (ensure it contains only 0s and 1s)
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension if not already present
        if len(img.shape) == 3:  # If shape is (H, W, D)
            img = np.expand_dims(img, axis=0)  # Add channel dimension -> (1, H, W, D)
        if len(mask.shape) == 3:
            mask = np.expand_dims(mask, axis=0)

        # Convert to tensors
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        return img, mask

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations shared across tensors."""

        flip_prob = self.augment_params.get("flip_prob", 0.5)
        rotate_prob = self.augment_params.get("rotate_prob", 0.3)
        noise_std = self.augment_params.get("noise_std", 0.0)
        intensity_shift = self.augment_params.get("intensity_shift", 0.0)
        intensity_scale = self.augment_params.get("intensity_scale", 0.0)

        # Random flips along spatial dimensions
        for dim in (1, 2, 3):
            if random.random() < flip_prob:
                img = torch.flip(img, dims=(dim,))
                mask = torch.flip(mask, dims=(dim,))

        # Random 90-degree rotations over random plane
        if random.random() < rotate_prob:
            axes = random.choice([(1, 2), (1, 3), (2, 3)])
            k = random.randint(0, 3)
            img = torch.rot90(img, k=k, dims=axes)
            mask = torch.rot90(mask, k=k, dims=axes)

        # Additive Gaussian noise
        if noise_std > 0:
            noise = torch.randn_like(img) * noise_std
            img = img + noise

        # Intensity shift/scale
        if intensity_shift > 0:
            shift = (random.random() * 2 - 1) * intensity_shift
            img = img + shift
        if intensity_scale > 0:
            scale = 1 + (random.random() * 2 - 1) * intensity_scale
            img = img * scale

        img = torch.clamp(img, 0.0, 1.0)

        return img, mask


def get_loaders(
    data_dir: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment: bool = True,
    augment_params: Optional[Dict[str, float]] = None,
):
    """Creates train and validation data loaders."""
    logger.info(f"Checking dataset path: {data_dir}")
    patients = check_dataset_structure(data_dir)

    # Split into train and validation sets
    train_patients, val_patients = train_test_split(patients, test_size=val_split, random_state=seed)

    # Create datasets
    train_dataset = MRIDataset(
        data_dir,
        patients=train_patients,
        augment=augment,
        augment_params=augment_params,
    )
    val_dataset = MRIDataset(data_dir, patients=val_patients, augment=False)

    # Create data loaders
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    return train_loader, val_loader