## Testing

Run the pytest suite to validate dataset loading and model forward pass:

```bash
pytest
```
# MRI-Segmentation

3D U-Net based pipeline for brain tumour segmentation from MRI volumes. The project includes
configurable training, evaluation utilities, deterministic preprocessing, and visualization tools.

## Project structure

```
├── config.py          # Dataclass + YAML loader for training configuration
├── data_loader.py     # Dataset utilities with deterministic and probabilistic augmentations
├── evaluate.py        # CLI for computing Dice/IoU metrics on saved checkpoints
├── model.py           # 3D U-Net implementation
├── tests/             # Pytest-based unit tests (data loader & model)
├── train.py           # Training script w/ mixed precision, early stopping, logging hooks
├── utils.py           # Metrics, early stopping helper, reproducibility utilities
├── visualize.py       # Slice-wise visualization of predictions vs. ground truth
├── requirements.txt   # Python dependencies
└── README.md
```

## Environment setup

1. Create and activate a virtual environment (Python ≥ 3.9 recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install GPU-enabled PyTorch following the [official instructions](https://pytorch.org/get-started/locally/).

## Dataset expectations

```
<data_dir>/
    patient_001/
        patient_001_t2.nii.gz
        patient_001_seg.nii.gz
    patient_002/
        patient_002_t2.nii.gz
        patient_002_seg.nii.gz
    ...
```

- `*_t2.nii.gz`: MRI volume.
- `*_seg.nii.gz`: binary tumour segmentation mask.

Update the `data_dir` field in `config.py` or in a YAML override to point to your dataset root.

## Configuration

Runtime settings live in `config.py` via a `TrainingConfig` dataclass. You can override values via
a YAML file and pass it with `--config path/to/config.yaml`. Unknown keys are stored under
`config.extra` so scripts can access custom metadata (e.g., restricted patient lists).

Key parameters:

- `data_dir`: dataset root.
- `batch_size`, `epochs`, `learning_rate`: training hyperparameters.
- `grad_accum_steps`: gradient accumulation steps.
- `dice_weight`: weighting between Dice and BCE losses.
- `checkpoint_path`: where the best model checkpoint is written.
- `threshold`: probability threshold for binarizing predictions.
- `patience`, `min_delta`: early-stopping settings.

## Training

```bash
python train.py --config configs/train.yaml  # optional config path
```

Features:

- Combined Dice + BCE loss with configurable weighting.
- Mixed precision (automatic when CUDA is available and enabled in config).
- Gradient accumulation for larger effective batch sizes.
- Reduce-on-plateau scheduler and early stopping.
- Deterministic seeding for reproducibility plus configurable augmentations (flip, rotation, noise,
  intensity jitter).
- Optional experiment tracking via TensorBoard (`enable_tensorboard`) and Weights & Biases
  (`enable_wandb`, `wandb_project`, etc.).
- Automatic checkpointing of the best validation model with stored config.

## Evaluation

Compute Dice and IoU metrics for a saved checkpoint:

```bash
python evaluate.py --checkpoint artifacts/best_model.pth --config configs/eval.yaml
```

By default the script uses configuration defaults; override batch size with `--batch-size` if needed.

## Visualization

Render sample slices comparing MRI, ground truth, and predictions:

```bash
python visualize.py --checkpoint artifacts/best_model.pth --num-samples 3 --num-slices 4
```

Use `--threshold` to override the binarization threshold for qualitative review.

## Reproducibility tips

- All scripts set seeds for Python, NumPy, and PyTorch; ensure you also fix seeds in any data
  preparation steps outside this project.
- Keep dataset splits consistent by reusing the same `seed` value in configuration files.
- Record the exact `config.as_dict()` stored in the checkpoint for experiment tracking.

## Next steps

- Integrate experiment tracking (TensorBoard, Weights & Biases).
- Add unit tests covering dataset and model forward pass.
- Extend augmentation space (elastic deformations, intensity shifts).