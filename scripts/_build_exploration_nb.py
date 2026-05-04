"""Generates notebooks/01_data_exploration.ipynb programmatically."""
import json
from pathlib import Path

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src if isinstance(src, list) else [src],
    }

def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src if isinstance(src, list) else [src],
    }

cells = []

# ── Cell 1: data directory guard ──────────────────────────────────────────────
cells.append(code(
"""import sys, os
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# nbconvert defaults cwd to the notebook's own directory; walk up to project root.
_cwd = Path.cwd()
if _cwd.name == "notebooks":
    os.chdir(_cwd.parent)

DATA_DIR = Path("data/raw/BraTS2020_TrainingData")

if not DATA_DIR.exists():
    print("\\nERROR: data directory not found:", DATA_DIR)
    print()
    print("To download the dataset:")
    print("  1. pip install kaggle")
    print("  2. Place your Kaggle API key at ~/.kaggle/kaggle.json")
    print("  3. kaggle datasets download awsaf49/brats20-dataset-training-validation")
    print("  4. Unzip so the layout is:")
    print("     data/raw/BraTS2020_TrainingData/BraTS20_Training_001/...")
    sys.exit(1)

patient_dirs = sorted(
    [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("BraTS20_Training_")]
)

if not patient_dirs:
    print("\\nERROR: No patient folders found in", DATA_DIR)
    print("Expected sub-directories named BraTS20_Training_NNN")
    sys.exit(1)

print(f"Found {len(patient_dirs)} patient folders")
patient = patient_dirs[0]
pid = patient.name
print(f"Using:  {pid}  (alphabetically first)")
"""
))

# ── Cell 2: load all 5 volumes ─────────────────────────────────────────────────
cells.append(code(
"""volumes = {}
for mod in ["t1", "t1ce", "t2", "flair", "seg"]:
    path = patient / f"{pid}_{mod}.nii"
    volumes[mod] = nib.load(str(path))

print("Loaded modalities:", list(volumes.keys()))
"""
))

# ── Cell 3: stats ──────────────────────────────────────────────────────────────
cells.append(code(
"""for mod in ["t1", "t1ce", "t2", "flair", "seg"]:
    img = volumes[mod]
    data = img.get_fdata(dtype=np.float32)
    aff = img.affine
    spacing = tuple(float(np.sqrt((aff[:3, i] ** 2).sum())) for i in range(3))

    print(f"\\n{'─'*52}")
    print(f"  {mod.upper()}")
    print(f"  shape        : {data.shape}")
    print(f"  dtype        : {img.get_data_dtype()}")
    print(f"  voxel spacing: {spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm")

    if mod == "seg":
        labels, counts = np.unique(data.astype(np.int32), return_counts=True)
        total = data.size
        print(f"  label distribution:")
        for lbl, cnt in zip(labels, counts):
            print(f"    label {lbl:>2d} : {cnt:>9,} voxels  ({100*cnt/total:6.3f}%)")
    else:
        fg = data[data > 0]
        print(f"  intensity min : {data.min():.2f}")
        print(f"  intensity max : {data.max():.2f}")
        print(f"  intensity mean: {fg.mean():.2f}  (foreground only, {len(fg):,} voxels)")
"""
))

# ── Cell 4: 4×3 modality grid ──────────────────────────────────────────────────
cells.append(code(
"""fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.suptitle(f"Patient {pid} — 4 modalities × 3 planes", fontsize=13, y=1.005)

for row, mod in enumerate(["t1", "t1ce", "t2", "flair"]):
    data = volumes[mod].get_fdata(dtype=np.float32)
    xi, yi, zi = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2

    slices = [
        (data[:, :, zi].T,  "Axial"),
        (data[:, yi, :].T,  "Coronal"),
        (data[xi, :, :].T,  "Sagittal"),
    ]
    for col, (sl, view) in enumerate(slices):
        ax = axes[row, col]
        p1, p99 = np.percentile(sl, 1), np.percentile(sl, 99)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="auto",
                  vmin=p1, vmax=p99)
        ax.set_title(f"{mod.upper()} — {view}", fontsize=9)
        ax.axis("off")

plt.tight_layout()
plt.savefig("notebooks/modalities_grid.png", dpi=120, bbox_inches="tight")
plt.show()
print("4×3 grid saved → notebooks/modalities_grid.png")
"""
))

# ── Cell 5: seg overlay on FLAIR ──────────────────────────────────────────────
cells.append(code(
"""flair = volumes["flair"].get_fdata(dtype=np.float32)
seg   = volumes["seg"].get_fdata(dtype=np.float32).astype(np.int32)

# Labels: 0=bg, 1=NCR/NET, 2=oedema, 4=enhancing
# Map to RGBA: index = label value (pad index 3 as transparent gap)
cmap_colors = [
    (0, 0, 0, 0),          # 0 background — transparent
    (1.0, 0.25, 0.25, 1),  # 1 NCR/NET    — red
    (0.25, 1.0, 0.25, 1),  # 2 oedema     — green
    (0, 0, 0, 0),          # 3 (unused)   — transparent
    (0.25, 0.5, 1.0, 1),   # 4 enhancing  — blue
]
seg_cmap = mcolors.ListedColormap(cmap_colors)

xi, yi, zi = flair.shape[0]//2, flair.shape[1]//2, flair.shape[2]//2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Patient {pid} — Segmentation overlay on FLAIR (α=0.4)", fontsize=12)

view_params = [
    (flair[:, :, zi].T, seg[:, :, zi].T, "Axial"),
    (flair[:, yi, :].T, seg[:, yi, :].T, "Coronal"),
    (flair[xi, :, :].T, seg[xi, :, :].T, "Sagittal"),
]

for ax, (fl_sl, sg_sl, title) in zip(axes, view_params):
    p1, p99 = np.percentile(fl_sl, 1), np.percentile(fl_sl, 99)
    ax.imshow(fl_sl, cmap="gray", origin="lower", aspect="auto", vmin=p1, vmax=p99)
    masked_seg = np.ma.masked_equal(sg_sl, 0)
    ax.imshow(masked_seg, cmap=seg_cmap, vmin=0, vmax=4,
              alpha=0.4, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

legend_elements = [
    Patch(facecolor=(1.0, 0.25, 0.25), label="1: NCR/NET (necrotic core)"),
    Patch(facecolor=(0.25, 1.0, 0.25), label="2: ED (oedema)"),
    Patch(facecolor=(0.25, 0.5,  1.0), label="4: ET (enhancing tumour)"),
]
axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8,
               framealpha=0.7)

plt.tight_layout()
plt.savefig("notebooks/seg_overlay.png", dpi=120, bbox_inches="tight")
plt.show()
print("Seg overlay saved → notebooks/seg_overlay.png")
"""
))

# ── Cell 6: per-modality intensity histograms ─────────────────────────────────
cells.append(code(
"""fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(
    f"Patient {pid} — Per-modality intensity histograms (foreground only, bg=0 masked)",
    fontsize=11,
)

for ax, mod in zip(axes, ["t1", "t1ce", "t2", "flair"]):
    data = volumes[mod].get_fdata(dtype=np.float32)
    fg = data[data > 0].ravel()
    ax.hist(fg, bins=120, color="steelblue", alpha=0.85, edgecolor="none")
    ax.set_title(mod.upper(), fontsize=10)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Voxel count")
    p1, p99 = np.percentile(fg, 1), np.percentile(fg, 99)
    ax.axvline(p1,  color="orange", lw=1.2, ls="--", label="p1")
    ax.axvline(p99, color="red",    lw=1.2, ls="--", label="p99")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("notebooks/intensity_histograms.png", dpi=120, bbox_inches="tight")
plt.show()
print("Histograms saved → notebooks/intensity_histograms.png")
"""
))

# ── Cell 7: markdown — label remapping ────────────────────────────────────────
cells.append(md(
"""## Label remapping and MONAI multi-channel conversion

The raw BraTS 2020 segmentation mask contains **four label values**:

| Label | Region | Abbreviation |
|-------|--------|--------------|
| 0 | Background | — |
| 1 | Necrotic core / non-enhancing tumour | NCR/NET |
| 2 | Peritumoral oedema | ED |
| **4** | **Enhancing tumour** | **ET** |

Label **4 must be remapped to 3** before any MONAI pipeline step, because
`monai.transforms.ConvertToMultiChannelBasedOnBratsClassesd` expects a compact
label set {0, 1, 2, 3} and interprets them as follows:

| Output channel | Name | Constituent labels (post-remap) |
|---|---|---|
| 0 | Whole Tumour (WT) | 1 + 2 + 3 |
| 1 | Tumour Core (TC) | 1 + 3 |
| 2 | Enhancing Tumour (ET) | 3 only |

**Transform order in the data pipeline:**
```python
monai.transforms.MapLabelValued(keys=["seg"], orig_labels=[4], target_labels=[3]),
monai.transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["seg"]),
```
These two transforms must appear in this order; applying `ConvertToMultiChannel...`
on the raw mask (with label 4 still present) silently produces a wrong ET channel.
"""
))

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Lumen (.venv)",
            "language": "python",
            "name": "lumen_venv",
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0",
        },
    },
    "cells": cells,
}

out = Path("notebooks/01_data_exploration.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written: {out}")
