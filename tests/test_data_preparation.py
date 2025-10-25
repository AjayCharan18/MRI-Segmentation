from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from config import TrainingConfig
from data_loader import check_dataset_structure
from data_preparation import prepare_dataset


def create_patient_case(root: Path, name: str, shape=(32, 32, 32)) -> None:
    patient_dir = root / name
    patient_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    image = np.random.rand(*shape).astype(np.float32)
    mask = (np.random.rand(*shape) > 0.5).astype(np.float32)
    nib.save(nib.Nifti1Image(image, affine), patient_dir / f"{name}_t2.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), patient_dir / f"{name}_seg.nii.gz")


def test_prepare_dataset_creates_synthetic_data(tmp_path: Path):
    target_dir = tmp_path / "dataset"
    config = TrainingConfig(
        data_dir=str(target_dir),
        dataset_url=None,
        synthetic_samples=2,
        synthetic_shape=(16, 16, 16),
    )

    prepare_dataset(config)

    patients = check_dataset_structure(str(target_dir))
    assert len(patients) == 2

    for patient in patients:
        img_path = target_dir / patient / f"{patient}_t2.nii.gz"
        mask_path = target_dir / patient / f"{patient}_seg.nii.gz"
        assert img_path.exists()
        assert mask_path.exists()
        img = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        assert img.shape == (16, 16, 16)
        assert mask.shape == (16, 16, 16)


def test_prepare_dataset_skips_when_data_exists(tmp_path: Path):
    target_dir = tmp_path / "dataset"
    create_patient_case(target_dir, "patient_001")

    config = TrainingConfig(
        data_dir=str(target_dir),
        dataset_url=None,
        synthetic_samples=5,
        synthetic_shape=(10, 10, 10),
    )

    prepare_dataset(config)

    patients = check_dataset_structure(str(target_dir))
    assert patients == ["patient_001"]

