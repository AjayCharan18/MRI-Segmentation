import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet
from data_loader import MRIDataset

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def dice_score(pred, target, smooth=1e-5):
    """Calculate Dice score between prediction and target."""
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def visualize_predictions(model, dataset, num_samples=3, num_slices=3, threshold=0.7):
    """
    Visualize multiple samples and slices from the dataset and classify tumor presence.
    
    Args:
        model: Trained model.
        dataset: Dataset object.
        num_samples: Number of samples to visualize.
        num_slices: Number of slices to visualize per sample.
        threshold: Threshold for binary predictions.
    """
    model.eval()

    for sample_idx in range(num_samples):
        img, mask = dataset[sample_idx]

        # Debug: Print image and mask shapes and values
        print(f"Sample {sample_idx + 1}")
        print(f"Image shape: {img.shape}, Min: {img.min()}, Max: {img.max()}")
        print(f"Mask shape: {mask.shape}, Min: {mask.min()}, Max: {mask.max()}")

        # Move data to the appropriate device
        img = img.unsqueeze(0).to(DEVICE)  # Add batch dimension
        mask = mask.to(DEVICE)

        # Get model prediction
        with torch.no_grad():
            raw_pred = model(img)  # Get raw model output
            print(f"Raw prediction: Min: {raw_pred.min()}, Max: {raw_pred.max()}")
            pred = torch.sigmoid(raw_pred)  # Apply sigmoid to get probabilities

            # Debug: Print histogram of raw predictions
            print(f"Raw prediction histogram: {torch.histc(raw_pred, bins=10, min=0, max=1)}")

            # Apply threshold to get binary predictions
            pred = (pred > threshold).float()

        # Check if tumor is present
        tumor_present = pred.sum() > 0  # If any pixel is above the threshold
        print(f"Tumor Present: {'Yes' if tumor_present else 'No'}")

        # Move data back to CPU for visualization
        img = img.cpu().squeeze().numpy()
        mask = mask.cpu().squeeze().numpy()
        pred = pred.cpu().squeeze().numpy()

        # Normalize the input image for better visualization
        img = (img - img.min()) / (img.max() - img.min())

        # Visualize multiple slices
        depth = img.shape[-1]
        slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)  # Get evenly spaced slices

        for slice_idx in slice_indices:
            img_slice = img[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]
            pred_slice = pred[:, :, slice_idx]

            # Calculate Dice score
            dice = dice_score(pred_slice, mask_slice)
            print(f"Slice {slice_idx + 1}, Dice Score: {dice:.4f}")

            # Debug: Print slice shapes and values
            print(f"Image slice shape: {img_slice.shape}, Min: {img_slice.min()}, Max: {img_slice.max()}")
            print(f"Mask slice shape: {mask_slice.shape}, Min: {mask_slice.min()}, Max: {mask_slice.max()}")
            print(f"Prediction slice shape: {pred_slice.shape}, Min: {pred_slice.min()}, Max: {pred_slice.max()}")

            # Plot the results
            plt.figure(figsize=(12, 4))

            # Input MRI
            plt.subplot(1, 3, 1)
            plt.title(f"Input MRI (Slice {slice_idx + 1})")
            plt.imshow(img_slice, cmap="gray")
            plt.axis("off")

            # Ground Truth
            plt.subplot(1, 3, 2)
            plt.title(f"Ground Truth (Slice {slice_idx + 1})")
            plt.imshow(mask_slice, cmap="gray")
            plt.axis("off")

            # Prediction
            plt.subplot(1, 3, 3)
            plt.title(f"Prediction (Slice {slice_idx + 1}, Tumor: {'Yes' if tumor_present else 'No'})")
            plt.imshow(pred_slice, cmap="binary")  # Use binary colormap for better visualization
            plt.axis("off")

            plt.show()

if __name__ == '__main__':
    # Load the trained model
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

    # Dataset directory
    DATASET_DIR = "C:/Users/Dell/Downloads/mri segmentation/data"

    # Filter out invalid entries (e.g., .DS_Store)
    patients = [p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p))]
    dataset = MRIDataset(DATASET_DIR, patients=patients)

    # Visualize multiple samples and slices
    visualize_predictions(model, dataset, num_samples=3, num_slices=3, threshold=0.7)  # Visualize 3 samples and 3 slices per sample