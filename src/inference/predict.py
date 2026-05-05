"""Single-case inference with sliding-window strategy.

Responsibilities:
- Accept 4 NIfTI paths (T1, T1ce, T2, FLAIR)
- Apply the same spatial preprocessing as val transforms
- Run sliding_window_inference and threshold logits
- Reconstruct multi-channel binary mask → single integer label volume
- Resample prediction back to original FLAIR voxel grid
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import nibabel.processing as nbp
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)


def predict_volume(
    t1_path: str | Path,
    t1ce_path: str | Path,
    t2_path: str | Path,
    flair_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    roi_size: Sequence[int] = (96, 96, 96),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Run sliding-window inference on one BraTS patient.

    Preprocessing mirrors ``get_val_transforms`` (Orientationd → Spacingd 1 mm
    → NormalizeIntensityd), and the prediction is resampled back to the
    original FLAIR voxel grid before returning.

    Returns
    -------
    seg_array : np.ndarray, dtype uint8
        Integer label volume in the original FLAIR image space.
        Label values: 0=background, 1=NCR/NET, 2=ED, 4=ET.
    original_affine : np.ndarray, shape (4, 4)
        Affine matrix from the FLAIR NIfTI file.
    original_header : nib.Nifti1Header
        Header from the FLAIR NIfTI file.
    """
    flair_path = Path(flair_path)

    # 1. Capture original geometry (FLAIR is the reference modality)
    orig_nib = nib.load(str(flair_path))
    original_affine = orig_nib.affine.copy()
    original_header = orig_nib.header.copy()

    # 2. Image-only preprocessing — same spatial ops as get_val_transforms
    preprocess = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys="image"),
    ])

    data = preprocess({
        "image": [str(t1_path), str(t1ce_path), str(t2_path), str(flair_path)]
    })

    # 3. Sliding-window inference
    image_tensor = data["image"].unsqueeze(0).to(device)  # (1, 4, H, W, D)
    model.eval()
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image_tensor,
            roi_size=list(roi_size),
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )

    preds = (torch.sigmoid(logits) > 0.5).float()[0].cpu().numpy()  # (3, H, W, D)

    # 4. Multi-channel [TC, WT, ET] → single integer label
    #    WT⊇TC⊇ET, so apply innermost last to maintain correct boundaries:
    #    background=0, ED=2 (WT only), NCR/NET=1 (TC \ ET), ET=4
    seg_proc = np.zeros(preds.shape[1:], dtype=np.uint8)
    seg_proc[preds[1] > 0.5] = 2   # WT region → ED label 2
    seg_proc[preds[0] > 0.5] = 1   # TC overwrites ED → NCR/NET label 1
    seg_proc[preds[2] > 0.5] = 4   # ET overwrites NCR → label 4

    # 5. Resample prediction back to original voxel space
    #    nibabel handles both the spacing change and axis permutation from Orientationd
    proc_affine = np.array(data["image"].meta["affine"], dtype=np.float64)
    if proc_affine.ndim == 3:
        proc_affine = proc_affine[0]

    seg_nib_proc = nib.Nifti1Image(seg_proc.astype(np.float32), proc_affine)
    seg_nib_orig = nbp.resample_from_to(seg_nib_proc, orig_nib, order=0, cval=0)
    seg_restored = np.round(seg_nib_orig.get_fdata()).astype(np.uint8)

    return seg_restored, original_affine, original_header
