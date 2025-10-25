import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import TrainingConfig, load_config
from data_loader import get_loaders
from data_preparation import prepare_dataset
from model import UNet
from utils import EarlyStopping, dice_coefficient, iou_score, set_seed, threshold_predictions

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss(
    logits: torch.Tensor, targets: torch.Tensor, dice_weight: float, smooth: float
) -> torch.Tensor:
    if not 0.0 <= dice_weight <= 1.0:
        raise ValueError("dice_weight must be between 0 and 1.")

    dice_component = dice_loss(logits, targets, smooth)
    if dice_weight == 1.0:
        return dice_component

    bce_component = F.binary_cross_entropy_with_logits(logits, targets)
    return dice_weight * dice_component + (1.0 - dice_weight) * bce_component


def evaluate(model: UNet, loader, device: str, config: TrainingConfig) -> Dict[str, float]:
    model.eval()
    if len(loader) == 0:
        raise ValueError("Validation loader is empty.")

    total_loss = 0.0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device, non_blocking=config.pin_memory)
            targets = targets.to(device, non_blocking=config.pin_memory)
            logits = model(data)
            loss = combined_loss(logits, targets, config.dice_weight, config.smooth)
            total_loss += loss.item()
            preds = threshold_predictions(logits, config.threshold)
            dice_scores.append(dice_coefficient(preds, targets).item())
            iou_scores.append(iou_score(preds, targets).item())

    return {
        "val_loss": total_loss / len(loader),
        "dice": sum(dice_scores) / len(dice_scores),
        "iou": sum(iou_scores) / len(iou_scores),
    }


def _init_tensorboard(config: TrainingConfig):
    if not config.enable_tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard not available; disable enable_tensorboard or install it.")
        return None

    log_dir = config.log_directory()
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _init_wandb(config: TrainingConfig) -> bool:
    if not config.enable_wandb:
        return False
    if wandb is None:
        print("wandb not installed; disable enable_wandb or install it.")
        return False

    wandb_kwargs = {
        "project": config.wandb_project,
        "entity": config.wandb_entity,
        "name": config.wandb_run_name,
        "config": config.as_dict(),
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    wandb.init(**wandb_kwargs)
    return True


def train(config: TrainingConfig) -> None:
    device = config.resolve_device()
    set_seed(config.seed)

    if config.auto_prepare_data:
        prepare_dataset(config)

    train_loader, val_loader = get_loaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        val_split=config.val_split,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        augment=config.enable_augmentation,
        augment_params={
            "flip_prob": config.augment_flip_prob,
            "rotate_prob": config.augment_rotate_prob,
            "noise_std": config.augment_noise_std,
            "intensity_shift": config.augment_intensity_shift,
            "intensity_scale": config.augment_intensity_scale,
        },
    )

    total_batches = len(train_loader)
    if total_batches == 0:
        raise ValueError("Training loader is empty. Check dataset path and contents.")

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    use_amp = config.mixed_precision and device.startswith("cuda")
    scaler = GradScaler(enabled=use_amp)
    writer = _init_tensorboard(config)
    wandb_enabled = _init_wandb(config)

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    group_size = config.grad_accum_steps

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        loop = tqdm(
            enumerate(train_loader),
            total=total_batches,
            desc=f"Epoch {epoch}/{config.epochs}",
        )

        for step, (data, targets) in loop:
            data = data.to(device, non_blocking=config.pin_memory)
            targets = targets.to(device, non_blocking=config.pin_memory)

            if step % config.grad_accum_steps == 0:
                remaining = total_batches - step
                group_size = min(config.grad_accum_steps, remaining)

            with autocast(enabled=use_amp):
                logits = model(data)
                batch_loss = combined_loss(logits, targets, config.dice_weight, config.smooth)
                loss = batch_loss / group_size

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += batch_loss.item()
            loop.set_postfix(train_loss=batch_loss.item())

            should_step = ((step + 1) % config.grad_accum_steps == 0) or ((step + 1) == total_batches)
            if should_step:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        avg_train_loss = epoch_loss / total_batches
        val_metrics = evaluate(model, val_loader, device, config)
        scheduler.step(val_metrics["val_loss"])

        log_payload = {
            "train_loss": avg_train_loss,
            **val_metrics,
            "epoch": epoch,
        }

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            writer.add_scalar("Dice/val", val_metrics["dice"], epoch)
            writer.add_scalar("IoU/val", val_metrics["iou"], epoch)

        if wandb_enabled:
            wandb.log(log_payload)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.as_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"Saved new best model to {checkpoint_path}")

        early_stopping.step(val_metrics["val_loss"])
        if early_stopping.should_stop:
            print("Early stopping triggered. Stopping training.")
            break

    if writer is not None:
        writer.close()
    if wandb_enabled:
        wandb.finish()

    print(f"Training finished. Best val loss: {best_val_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MRI segmentation model.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    training_config = load_config(args.config)
    train(training_config)