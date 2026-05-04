"""Training loop.

Responsibilities:
- Load config, build dataloaders, model, loss, optimizer, scheduler
- AMP-wrapped forward/backward pass
- Sliding-window validation every val_frequency epochs
- Checkpoint saving (best val Dice mean across WT/TC/ET)
- Optional Weights & Biases logging
"""
