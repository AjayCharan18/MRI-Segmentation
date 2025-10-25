import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import load_config
from data_loader import MRIDataset
from model import UNet
from utils import dice_coefficient, threshold_predictions


def visualize_predictions(
    model: UNet,
    dataset: MRIDataset,
    device: str,
    num_samples: int,
    num_slices: int,
    threshold: float,
) -> None:
    """Visualize predictions and ground-truth masks for a subset of samples."""

    model.eval()

    for sample_idx in range(min(num_samples, len(dataset))):
        img, mask = dataset[sample_idx]
        img = img.unsqueeze(0).to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits = model(img)
            preds = threshold_predictions(logits, threshold)

        tumor_present = preds.sum() > 0

        img_np = img.cpu().squeeze().numpy()
        mask_np = mask.cpu().squeeze().numpy()
        pred_np = preds.cpu().squeeze().numpy()

        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        depth = img_np.shape[-1]
        slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

        for slice_idx in slice_indices:
            img_slice = img_np[:, :, slice_idx]
            mask_slice = mask_np[:, :, slice_idx]
            pred_slice = pred_np[:, :, slice_idx]

            dice = dice_coefficient(
                torch.tensor(pred_slice).unsqueeze(0).unsqueeze(0),
                torch.tensor(mask_slice).unsqueeze(0).unsqueeze(0),
            ).item()

            plt.figure(figsize=(12, 4))
            plt.suptitle(
                f"Sample {sample_idx + 1} | Slice {slice_idx + 1} | Tumor: {'Yes' if tumor_present else 'No'} | Dice: {dice:.3f}"
            )

            plt.subplot(1, 3, 1)
            plt.title("Input MRI")
            plt.imshow(img_slice, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(mask_slice, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(pred_slice, cmap="binary")
            plt.axis("off")

            plt.tight_layout()
            plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MRI segmentation predictions.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of volumes to visualize.")
    parser.add_argument("--num-slices", type=int, default=3, help="Number of slices per volume.")
    parser.add_argument("--threshold", type=float, default=None, help="Prediction threshold override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = config.resolve_device()

    checkpoint_path = args.checkpoint or config.checkpoint_path
    if not checkpoint_path:
        raise ValueError("Checkpoint path must be provided via --checkpoint or config file.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)

    model = UNet().to(device)
    model.load_state_dict(state_dict)

    threshold = args.threshold if args.threshold is not None else config.threshold

    patients = sorted(config.extra.get("patients", [])) or None
    if patients is None:
        patients = [
            entry
            for entry in os.listdir(config.data_dir)
            if os.path.isdir(os.path.join(config.data_dir, entry))
        ]

    dataset = MRIDataset(config.data_dir, patients=patients, augment=False)

    visualize_predictions(
        model=model,
        dataset=dataset,
        device=device,
        num_samples=args.num_samples,
        num_slices=args.num_slices,
        threshold=threshold,
    )


if __name__ == "__main__":
    main()