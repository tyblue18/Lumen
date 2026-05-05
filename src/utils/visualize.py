"""Visualization helpers for BraTS segmentation evaluation.

- overlay_segmentation: blend multi-channel binary mask onto a grayscale MRI slice
- build_qualitative_grid: N-patient × 3-column figure (FLAIR / GT / Pred)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


# Channel order emitted by ConvertToMultiChannelBasedOnBratsClassesd: [TC, WT, ET]
# Render WT first (largest region, underneath), TC next, ET on top (smallest).
_RENDER_ORDER = [1, 0, 2]          # WT → TC → ET

# (R, G, B) colours for each channel index
_REGION_COLORS: dict[int, tuple[int, int, int]] = {
    0: (30,  144, 255),  # TC — dodger blue
    1: (50,  205,  50),  # WT — lime green
    2: (220,  20,  60),  # ET — crimson
}

_REGION_LABELS = {0: "TC", 1: "WT", 2: "ET"}


def _contrast_stretch(arr: np.ndarray) -> np.ndarray:
    """Stretch arr to [0, 1] using the 1st–99th percentile range."""
    p1, p99 = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    if p99 <= p1:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - p1) / (p99 - p1), 0.0, 1.0)


def overlay_segmentation(
    image_slice: np.ndarray,
    seg_slice: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a multi-channel binary segmentation mask on a grayscale MRI slice.

    Args:
        image_slice: ``(H, W)`` float array — any MRI modality.
                     Contrast-stretched to [p1, p99] before compositing.
        seg_slice:   ``(3, H, W)`` binary float/bool array with channels
                     ``[TC, WT, ET]``, as produced by
                     ``ConvertToMultiChannelBasedOnBratsClassesd``.
        alpha:       Opacity of the colour overlay (0 = invisible, 1 = opaque).

    Returns:
        RGB :class:`PIL.Image.Image`.
        Colour coding: **WT = green**, **TC = blue**, **ET = red**.
        Regions are blended in order WT → TC → ET so that smaller structures
        always appear on top.
    """
    H, W = image_slice.shape[-2], image_slice.shape[-1]

    # ── Base grayscale layer ─────────────────────────────────────────────
    gray = _contrast_stretch(image_slice.squeeze())   # (H, W) in [0, 1]
    canvas = np.stack([gray * 255.0] * 3, axis=-1)    # (H, W, 3) float32

    # ── Blend each region ────────────────────────────────────────────────
    for ch in _RENDER_ORDER:
        mask = seg_slice[ch].astype(bool)
        if not mask.any():
            continue
        color = np.array(_REGION_COLORS[ch], dtype=np.float32)
        canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * color

    return Image.fromarray(canvas.clip(0, 255).astype(np.uint8), mode="RGB")


def build_qualitative_grid(
    rows: list[dict],
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Build and save an N×3 qualitative grid.

    Each row is one patient; columns are FLAIR (grayscale), GT overlay,
    Prediction overlay.

    Args:
        rows: list of dicts, each with keys:
              ``patient_id`` (str),
              ``flair``      (H, W) float array,
              ``gt``         (3, H, W) binary array,
              ``pred``       (3, H, W) binary array.
        output_path: where to write the PNG.
        dpi: output resolution (150 gives ~2100-px wide for 14-inch figure).

    Returns:
        Resolved ``Path`` to the saved file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(14, n * 4.5),
                              squeeze=False)

    col_titles = ["FLAIR (mid-axial)", "Ground truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=6)

    for row_i, data in enumerate(rows):
        flair = data["flair"]
        gt    = data["gt"]
        pred  = data["pred"]
        pid   = data["patient_id"]

        gray_norm = _contrast_stretch(flair)

        # Col 0 — FLAIR
        ax0 = axes[row_i, 0]
        ax0.imshow(gray_norm.T, cmap="gray", origin="lower", aspect="auto")
        ax0.set_ylabel(pid, fontsize=8, labelpad=4)
        ax0.axis("off")

        # Col 1 — GT overlay
        axes[row_i, 1].imshow(
            np.array(overlay_segmentation(flair, gt)).transpose(1, 0, 2),
            origin="lower", aspect="auto",
        )
        axes[row_i, 1].axis("off")

        # Col 2 — Pred overlay
        axes[row_i, 2].imshow(
            np.array(overlay_segmentation(flair, pred)).transpose(1, 0, 2),
            origin="lower", aspect="auto",
        )
        axes[row_i, 2].axis("off")

    legend_patches = [
        Patch(facecolor=tuple(c / 255 for c in _REGION_COLORS[ch]),
              label=_REGION_LABELS[ch])
        for ch in (1, 0, 2)   # WT, TC, ET — display order
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=11,
        framealpha=0.85,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out.resolve()
