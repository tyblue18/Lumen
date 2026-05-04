"""Evaluation metrics for BraTS segmentation.

Channel order from ConvertToMultiChannelBasedOnBratsClassesd (verified in MONAI 1.5.2 source):
  ch 0 = TC  (labels 1 | 4  — necrotic core + enhancing tumour)
  ch 1 = WT  (labels 1 | 2 | 4 — whole tumour)
  ch 2 = ET  (label 4 — enhancing tumour only)

RegionDice maps these indices to named regions explicitly so a transposition
bug would surface as obviously-wrong per-region numbers rather than silent
average corruption.
"""

import torch
from monai.metrics import DiceMetric


class RegionDice:
    """Accumulates per-region Dice over a validation epoch.

    Usage::

        metric = RegionDice()
        for batch in val_loader:
            preds = sliding_window_inference(...)
            preds_bin = (torch.sigmoid(preds) > 0.5).float()
            metric.update(preds_bin, batch["label"])
        scores = metric.compute()   # {"TC": float, "WT": float, "ET": float}
        metric.reset()
    """

    # Channel order as produced by ConvertToMultiChannelBasedOnBratsClassesd.
    # Verified from MONAI 1.5.2 source: result = [TC, WT, ET].
    _CHANNEL_NAMES = ("TC", "WT", "ET")

    def __init__(self) -> None:
        # include_background=True because all 3 channels are foreground regions;
        # reduction="mean_batch" returns one Dice per channel averaged over the batch.
        self._metric = DiceMetric(include_background=True, reduction="mean_batch")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch.  Both tensors must be binary float (B, 3, ...)."""
        self._metric(y_pred=preds, y=targets)

    def compute(self) -> dict[str, float]:
        """Return ``{"TC": ..., "WT": ..., "ET": ...}`` and reset the accumulator."""
        scores = self._metric.aggregate()   # shape (3,) — one value per channel
        self._metric.reset()
        return {
            name: float(scores[i])
            for i, name in enumerate(self._CHANNEL_NAMES)
        }

    def reset(self) -> None:
        self._metric.reset()
