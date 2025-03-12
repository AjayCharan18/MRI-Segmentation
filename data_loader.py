import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, RandFlip, RandRotate, RandZoom, ToTensor
from sklearn.model_selection import train_test_split
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dataset_structure(data_dir):
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
    
    valid_patients = []
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
    def __init__(self, data_dir: str, patients: list, transform=None):
        self.data_dir = data_dir
        self.patients = patients
        self.transform = transform

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
        img = resize(img, (32, 32, 32), order=1, preserve_range=True)  # Resize to 32x32x32
        mask = resize(mask, (32, 32, 32), order=0, preserve_range=True)

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

        # Apply transforms
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask

def get_loaders(data_dir: str, batch_size: int = 8, val_split: float = 0.2, seed: int = 42):
    """Creates train and validation data loaders."""
    logger.info(f"Checking dataset path: {data_dir}")
    patients = check_dataset_structure(data_dir)

    # Split into train and validation sets
    train_patients, val_patients = train_test_split(patients, test_size=val_split, random_state=seed)

    # Define transforms
    train_transforms = Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandRotate(range_x=15, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ToTensor()
    ])
    val_transforms = Compose([
        ToTensor()
    ])

    # Create datasets
    train_dataset = MRIDataset(data_dir, patients=train_patients, transform=train_transforms)
    val_dataset = MRIDataset(data_dir, patients=val_patients, transform=val_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader