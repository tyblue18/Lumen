"""Smoke-test the environment and data layout before training.

Checks:
- Python / PyTorch / MONAI / CUDA availability
- data_dir exists and contains the expected folder structure
- One patient can be loaded and transformed without errors
- Model instantiates and forward-passes a random patch
"""
