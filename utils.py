from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed across libraries to improve reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute Dice coefficient between prediction and target tensors."""

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Compute Intersection over Union (IoU) for segmentation."""

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    total = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    union = total - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    best_score: float = float("inf")
    counter: int = 0
    should_stop: bool = False

    def step(self, metric: float) -> None:
        if metric < self.best_score - self.min_delta:
            self.best_score = metric
            self.counter = 0
            self.should_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def threshold_predictions(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply sigmoid + threshold to logits to obtain binary predictions."""

    return (torch.sigmoid(logits) > threshold).float()
