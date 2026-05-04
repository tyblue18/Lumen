"""Evaluation metrics for BraTS segmentation.

- Per-region Dice (WT, TC, ET) via monai.metrics.DiceMetric
- HD95 (95th-percentile Hausdorff distance) via monai.metrics.HausdorffDistanceMetric
- Metrics are computed on full volumes after sliding-window inference, not patches
"""
