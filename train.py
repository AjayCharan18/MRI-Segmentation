import torch
import torch.optim as optim
from tqdm import tqdm
from model import UNet
from data_loader import get_loaders
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-4
EPOCHS = 20  # Increased epochs for better training
DATASET_DIR = "C:/Users/Dell/Downloads/mri segmentation/data"
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4

# Dice Loss Function
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Training Function
def train():
    # Initialize model, optimizer, and scheduler
    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Initialize GradScaler for mixed precision training (only for CUDA)
    scaler = GradScaler() if DEVICE == 'cuda' else None

    # Get data loaders
    train_loader, val_loader = get_loaders(DATASET_DIR, batch_size=BATCH_SIZE)

    # Track best validation loss
    best_loss = float('inf')

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        optimizer.zero_grad()

        for i, (data, targets) in enumerate(loop):
            data, targets = data.to(DEVICE), targets.to(DEVICE)

            if DEVICE == 'cuda':
                # Mixed precision training (only for CUDA)
                with autocast():
                    predictions = model(data)
                    loss = dice_loss(predictions, targets) / GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
            else:
                # Standard training (for CPU)
                predictions = model(data)
                loss = dice_loss(predictions, targets) / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

            # Gradient accumulation
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if DEVICE == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Update loss tracking
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            loop.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION_STEPS)

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                predictions = model(data)
                val_loss += dice_loss(predictions, targets).item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} completed. Avg Val Loss: {avg_val_loss:.4f}")

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Best model saved with val loss: {best_loss:.4f}")

# Visualization Function
def visualize_predictions(model, dataset, num_samples=3, num_slices=3, threshold=0.5):  # Adjusted threshold
    model.eval()
    for sample_idx in range(num_samples):
        img, mask = dataset[sample_idx]
        img = img.unsqueeze(0).to(DEVICE)
        mask = mask.to(DEVICE)

        with torch.no_grad():
            pred = torch.sigmoid(model(img))
            pred = (pred > threshold).float()

        img = img.cpu().squeeze().numpy()
        mask = mask.cpu().squeeze().numpy()
        pred = pred.cpu().squeeze().numpy()

        depth = img.shape[-1]
        slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

        for slice_idx in slice_indices:
            img_slice = img[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]
            pred_slice = pred[:, :, slice_idx]

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.title(f"Input MRI (Slice {slice_idx + 1})")
            plt.imshow(img_slice, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title(f"Ground Truth (Slice {slice_idx + 1})")
            plt.imshow(mask_slice, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Prediction (Slice {slice_idx + 1})")
            plt.imshow(pred_slice, cmap="binary")
            plt.axis("off")

            plt.show()

# Main entry point
if __name__ == '__main__':
    train()