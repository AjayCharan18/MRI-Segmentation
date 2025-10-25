from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml


@dataclass(slots=True)
class TrainingConfig:
    """Configuration container for training and evaluation."""

    data_dir: str = "C:/Users/Dell/Downloads/mri_segmentation"
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 20
    grad_accum_steps: int = 4
    num_workers: int = 4
    val_split: float = 0.2
    seed: int = 42
    patience: int = 5
    min_delta: float = 1e-4
    checkpoint_path: str = "artifacts/best_model.pth"
    device: Optional[str] = None
    dice_weight: float = 0.5
    threshold: float = 0.5
    smooth: float = 1e-5
    pin_memory: bool = True
    mixed_precision: bool = True
    enable_augmentation: bool = True
    augment_flip_prob: float = 0.5
    augment_rotate_prob: float = 0.3
    augment_noise_std: float = 0.05
    augment_intensity_shift: float = 0.1
    augment_intensity_scale: float = 0.1
    enable_tensorboard: bool = True
    log_dir: str = "runs"
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    dataset_url: Optional[str] = None
    auto_prepare_data: bool = True
    synthetic_samples: int = 4
    synthetic_shape: Tuple[int, int, int] = (32, 32, 32)

    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def log_directory(self) -> Path:
        return Path(self.log_dir)

    def as_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data.pop("extra", None)
        return data


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load configuration from YAML file or use defaults."""

    if config_path is None:
        return TrainingConfig()

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found")

    with config_file.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}

    known_fields = {field.name for field in dataclasses.fields(TrainingConfig)}
    config_kwargs: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}

    for key, value in payload.items():
        if key in known_fields:
            config_kwargs[key] = value
        else:
            extras[key] = value

    config = TrainingConfig(**config_kwargs)
    config.extra.update(extras)
    return config
