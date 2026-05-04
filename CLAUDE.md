# Lumen — BraTS 2020 Brain-Tumour Segmentation

## Project overview

3-D patch-based segmentation of glioma sub-regions using a MONAI UNet trained on
BraTS 2020 multi-modal MRI. Outputs three binary masks (sigmoid, not softmax) for
the three overlapping tumour regions: whole tumour (WT), tumour core (TC), and
enhancing tumour (ET).

## Repository layout

```
configs/        YAML hyper-parameter files (default.yaml is the reference)
data/
  raw/          Raw NIfTI volumes — never modified by training code
notebooks/      Exploratory / analysis notebooks
runs/           Training artefacts (checkpoints, logs) — git-ignored
scripts/        CLI entry points: train.py, evaluate.py, predict.py, verify_setup.py
src/
  models/       Model definitions (unet.py wraps monai.networks.nets.UNet)
  training/     Trainer, losses, metrics
  inference/    Sliding-window predict
  utils/        Visualisation helpers
```

## Environment

- Python venv at `.venv/` — activate with `.venv\Scripts\activate` on Windows
- PyTorch installed from the CUDA 12.8 index (`+cu128` wheels); requires CUDA-capable GPU
- Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
  then `pip install -r requirements.txt`
- Smoke-test: `python scripts/verify_setup.py` — must exit 0 before running anything else

## Model

| Parameter       | Value                          |
|-----------------|-------------------------------|
| Architecture    | `monai.networks.nets.UNet`    |
| `in_channels`   | 4 (T1, T1ce, T2, FLAIR)       |
| `out_channels`  | 3 (WT, TC, ET)                |
| `channels`      | (32, 64, 128, 256, 320)       |
| `strides`       | (2, 2, 2, 2)                  |
| `num_res_units` | 2                             |
| Activation      | Sigmoid per channel           |
| Loss            | `DiceCELoss(sigmoid=True)`    |

Regions are nested/overlapping (WT ⊇ TC ⊇ ET) — sigmoid per channel is correct,
softmax would be wrong here.

## Training

```
python scripts/train.py --config configs/default.yaml --run-dir runs/exp1
```

Key config knobs (`configs/default.yaml`):

| Key                      | Default           |
|--------------------------|-------------------|
| `patch_size`             | [96, 96, 96]      |
| `samples_per_volume`     | 2                 |
| `batch_size`             | 2                 |
| `epochs`                 | 100               |
| `optimizer.lr`           | 1e-4 (AdamW)      |
| `scheduler`              | CosineAnnealingLR |
| `amp`                    | true              |
| `sliding_window_overlap` | 0.5               |

Validation metric: mean Dice across WT / TC / ET (sliding-window inference).
Best checkpoint saved to `{run_dir}/best_model.pt`.

## Dataset: BraTS 2020

### File suffixes

| Modality / label  | Filename pattern      |
|-------------------|-----------------------|
| T1                | `*_t1.nii`            |
| T1ce              | `*_t1ce.nii`          |
| T2                | `*_t2.nii`            |
| FLAIR             | `*_flair.nii`         |
| Segmentation mask | `*_seg.nii`           |

Segmentation label values: **0** background, **1** necrotic core (NCR/NET),
**2** peritumoral oedema (ED), **4** enhancing tumour (ET).
Use `monai.transforms.ConvertToMultiChannelBasedOnBratsClassesd` directly on the
raw labels — it checks for label 4 internally.
**Do not** remap 4 → 3; that silently zeros the ET channel.

### Expected layout

```
data/raw/BraTS2020_TrainingData/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii
│   ├── BraTS20_Training_001_t1.nii
│   ├── BraTS20_Training_001_t1ce.nii
│   ├── BraTS20_Training_001_t2.nii
│   └── BraTS20_Training_001_seg.nii
├── BraTS20_Training_002/
│   └── ...
...
```

### This run's specifics

- **Source:** Kaggle mirror `awsaf49/brats20-dataset-training-validation`
- **Path:** `data/raw/BraTS2020_TrainingData/`
- **369 patient folders** named `BraTS20_Training_001` through `BraTS20_Training_369`
- Files are `.nii` (uncompressed); glob patterns must use `*.nii`, not `*.nii.gz`
- Top level also contains `name_mapping.csv` and `survival_info.csv` —
  the patient-folder scanner must filter to directories only, or match the pattern
  `BraTS20_Training_*`

## Multi-channel label conversion

`ConvertToMultiChannelBasedOnBratsClassesd` (MONAI 1.3+) inspects raw BraTS label
values {0, 1, 2, 4} and produces three binary channels:

| Channel | Region | Labels included |
|---------|--------|-----------------|
| 0 | TC — Tumour Core | 1 \| 4 |
| 1 | WT — Whole Tumour | 1 \| 2 \| 4 |
| 2 | ET — Enhancing Tumour | 4 |

Apply the transform directly after `EnsureChannelFirstd`; no label remapping is
needed or correct.
