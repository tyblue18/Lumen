"""Run inference on a single BraTS patient and write a segmentation NIfTI.

Usage:
    python scripts/predict.py \\
        --t1    data/raw/BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii \\
        --t1ce  data/raw/BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii \\
        --t2    data/raw/BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t2.nii \\
        --flair data/raw/BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii \\
        --ckpt  runs/brats_v1/best.pt \\
        --output runs/brats_v1/predictions/test_001/seg.nii.gz \\
        --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predict import predict_volume
from src.models.unet import build_unet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS single-case prediction")
    p.add_argument("--t1",       required=True, help="T1 NIfTI path")
    p.add_argument("--t1ce",     required=True, help="T1ce NIfTI path")
    p.add_argument("--t2",       required=True, help="T2 NIfTI path")
    p.add_argument("--flair",    required=True, help="FLAIR NIfTI path")
    p.add_argument("--ckpt",     required=True, help="Checkpoint .pt path")
    p.add_argument("--output",   required=True,
                   help="Output segmentation NIfTI, e.g. runs/.../seg.nii.gz")
    p.add_argument("--config",   default="configs/default.yaml")
    p.add_argument("--visualize", action="store_true",
                   help="Also save a mid-axial overlay PNG next to the NIfTI")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_stats(seg: np.ndarray) -> None:
    labels, counts = np.unique(seg, return_counts=True)
    names = {0: "BG", 1: "NCR/NET", 2: "ED", 4: "ET"}
    for lbl, cnt in zip(labels, counts):
        pct = 100.0 * cnt / seg.size
        print(f"  label {int(lbl)}  ({names.get(int(lbl), f'lbl{lbl}'):8s}): "
              f"{cnt:>9,} voxels  ({pct:.2f} %)")


def _save_overlay(flair_path: str, seg: np.ndarray, output_path: Path) -> None:
    """Write a mid-axial FLAIR + prediction overlay PNG."""
    from src.utils.visualize import overlay_segmentation

    flair_nib = nib.load(str(flair_path))
    flair_vol = flair_nib.get_fdata(dtype=np.float32)  # (H, W, D)

    d = flair_vol.shape[-1] // 2
    flair_slice = flair_vol[:, :, d]   # (H, W)

    # Single-label → 3-channel binary [TC, WT, ET] for overlay
    seg_d = seg[:, :, d]
    seg_slice = np.stack([
        ((seg_d == 1) | (seg_d == 4)).astype(np.float32),   # TC  ch0
        ((seg_d == 1) | (seg_d == 2) | (seg_d == 4)).astype(np.float32),  # WT ch1
        (seg_d == 4).astype(np.float32),                    # ET  ch2
    ], axis=0)  # (3, H, W)

    img = overlay_segmentation(flair_slice, seg_slice, alpha=0.4)
    png_path = output_path.parent / "seg_overlay.png"
    img.save(str(png_path))
    print(f"Saved overlay  -> {png_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Config
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    roi_size = cfg.get("sliding_window_roi", [96, 96, 96])
    overlap  = float(cfg.get("sliding_window_overlap", 0.5))

    # Device + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    model = build_unet(cfg.get("model")).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded  : {args.ckpt}  "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"best_dice={ckpt.get('best_dice', float('nan')):.4f})")

    # Inference
    print("\nRunning sliding-window inference …")
    seg, affine, header = predict_volume(
        t1_path=args.t1,
        t1ce_path=args.t1ce,
        t2_path=args.t2,
        flair_path=args.flair,
        model=model,
        device=device,
        roi_size=roi_size,
        overlap=overlap,
    )

    print(f"Seg shape : {seg.shape}  dtype={seg.dtype}")
    _label_stats(seg)

    # Save segmentation NIfTI
    seg_nib = nib.Nifti1Image(seg.astype(np.int16), affine, header)
    nib.save(seg_nib, str(output_path))
    print(f"\nSaved seg  -> {output_path}")

    # Optional mid-axial overlay
    if args.visualize:
        _save_overlay(args.flair, seg, output_path)


if __name__ == "__main__":
    main()
