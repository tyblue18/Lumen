"""Unit tests for dataset and transform pipeline.

Tests:
- Patient discovery and train/val split reproducibility
- Label remapping (4 → 3) and multi-channel conversion
- Per-channel non-zero normalization
- Patch shape after random crop
- No data leakage between train and val splits
"""
