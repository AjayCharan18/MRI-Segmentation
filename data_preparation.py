from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple
from urllib.request import urlretrieve

import nibabel as nib
import numpy as np

from config import TrainingConfig
from data_loader import check_dataset_structure

logger = logging.getLogger(__name__)


def _has_dataset(data_dir: Path) -> bool:
    try:
        check_dataset_structure(str(data_dir))
        return True
    except Exception:
        return False


def _download_dataset(url: str, target_dir: Path) -> None:
    logger.info("Downloading dataset from %s", url)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        urlretrieve(url, tmp_path)
        logger.info("Download complete. Extracting contents...")
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(target_dir)
        elif tarfile.is_tarfile(tmp_path):
            with tarfile.open(tmp_path, "r:*") as tf:
                tf.extractall(target_dir)
        else:
            raise ValueError("Unsupported archive format. Provide .zip or .tar.* file.")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _create_synthetic_sample(patient_dir: Path, shape: Sequence[int]) -> None:
    image = np.random.rand(*shape).astype(np.float32)
    mask = (np.random.rand(*shape) > 0.6).astype(np.float32)
    affine = np.eye(4)

    nib.save(nib.Nifti1Image(image, affine), patient_dir / f"{patient_dir.name}_t2.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), patient_dir / f"{patient_dir.name}_seg.nii.gz")


def _create_synthetic_dataset(data_dir: Path, samples: int, shape: Tuple[int, int, int]) -> None:
    logger.info("Creating synthetic dataset with %d samples at %s", samples, data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(samples):
        patient_name = f"synthetic_{idx:03d}"
        patient_dir = data_dir / patient_name
        patient_dir.mkdir(exist_ok=True)
        _create_synthetic_sample(patient_dir, shape)


def prepare_dataset(config: TrainingConfig) -> None:
    data_dir = Path(config.data_dir)

    if _has_dataset(data_dir):
        logger.info("Dataset found at %s", data_dir)
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    if config.dataset_url:
        try:
            _download_dataset(config.dataset_url, data_dir)
            if _has_dataset(data_dir):
                logger.info("Dataset downloaded and ready at %s", data_dir)
                return
            logger.warning(
                "Downloaded data from %s but expected structure was not detected. Falling back to synthetic data.",
                config.dataset_url,
            )
        except Exception as exc:
            logger.warning("Failed to download dataset from %s: %s", config.dataset_url, exc)

    _create_synthetic_dataset(data_dir, config.synthetic_samples, config.synthetic_shape)
    logger.info("Synthetic dataset prepared at %s", data_dir)

