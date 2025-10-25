from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader

from config import TrainingConfig, load_config
from data_loader import MRIDataset
from model import UNet
from utils import dice_coefficient, iou_score, threshold_predictions


def evaluate_model(config: TrainingConfig, checkpoint_path: str, batch_size: int | None = None) -> Dict[str, float]:
    device = config.resolve_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)

    model = UNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    patients = sorted(config.extra.get("patients", [])) or [
        entry
        for entry in os.listdir(config.data_dir)
        if os.path.isdir(os.path.join(config.data_dir, entry))
    ]

    dataset = MRIDataset(config.data_dir, patients=patients, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size or config.batch_size, shuffle=False)

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            logits = model(data)
            preds = threshold_predictions(logits, config.threshold)

            dice_scores.append(dice_coefficient(preds, targets).item())
            iou_scores.append(iou_score(preds, targets).item())

    return {
        "dice": sum(dice_scores) / len(dice_scores),
        "iou": sum(iou_scores) / len(iou_scores),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MRI segmentation model.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size to use during evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics = evaluate_model(config, args.checkpoint, batch_size=args.batch_size)
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
