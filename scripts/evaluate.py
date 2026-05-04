"""Evaluate a checkpoint on the validation set and report per-region Dice.

Usage:
    python scripts/evaluate.py --ckpt runs/default/best_model.pth [--config configs/default.yaml]

Outputs a Dice table (WT / TC / ET) and saves a qualitative overlay figure.
"""
