from pathlib import Path

import os
import numpy as np
import nibabel as nib
import pytest
import torch

from data_loader import MRIDataset


@pytest.fixture
def synthetic_dataset(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()

    affine = np.eye(4)

    for idx in range(2):
        patient_dir = root / f"patient_{idx:03d}"
        patient_dir.mkdir()
        image = np.random.rand(32, 32, 32).astype(np.float32)
        mask = (np.random.rand(32, 32, 32) > 0.5).astype(np.float32)

        nib.save(nib.Nifti1Image(image, affine), patient_dir / f"patient_{idx:03d}_t2.nii.gz")
        nib.save(nib.Nifti1Image(mask, affine), patient_dir / f"patient_{idx:03d}_seg.nii.gz")

    return root


def test_dataset_returns_tensor_pair(synthetic_dataset: Path):
    patients = sorted(os.listdir(synthetic_dataset))
    dataset = MRIDataset(
        str(synthetic_dataset),
        patients=patients,
        augment=True,
        augment_params={
            "flip_prob": 1.0,
            "rotate_prob": 1.0,
            "noise_std": 0.1,
            "intensity_shift": 0.1,
            "intensity_scale": 0.1,
        },
    )

    img, mask = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape == (1, 32, 32, 32)
    assert mask.shape == (1, 32, 32, 32)
    assert img.min() >= 0.0 and img.max() <= 1.0
    unique_vals = sorted(mask.unique().tolist())
    assert unique_vals in ([0.0], [0.0, 1.0], [1.0])


def test_dataset_len_matches_patients(synthetic_dataset: Path):
    patients = sorted(os.listdir(synthetic_dataset))
    dataset = MRIDataset(str(synthetic_dataset), patients=patients)

    assert len(dataset) == len(patients)


def test_dataset_raises_for_missing_files(tmp_path: Path):
    dataset = MRIDataset(str(tmp_path), patients=["missing"], augment=False)
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]
