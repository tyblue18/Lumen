"""Entry point: train the 3D U-Net on BraTS data.

Usage:
    python scripts/train.py [--config configs/default.yaml] [--run-name smoke]
                            [--resume runs/smoke/last.pt] [--subset data/raw/subset/]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import DataLoader

# Allow `python scripts/train.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import get_brats_datalist, make_dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet import build_unet
from src.training.losses import build_loss
from src.training.metrics import RegionDice
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True trades strict reproducibility for speed;
    # weights are still reproducibly initialised via the manual seeds above.
    torch.backends.cudnn.benchmark = True


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _guard_splits(train_list: list, val_list: list) -> tuple[list, list]:
    """Ensure neither split is empty — needed when data dir has very few patients."""
    if not train_list and not val_list:
        raise ValueError("Datalist is empty — check data_dir path.")
    if not val_list:
        print("WARNING: val split is empty (too few patients for hash split). "
              "Using last train patient as val.")
        val_list = [train_list[-1]]
        train_list = train_list[:-1]
    if not train_list:
        print("WARNING: train split is empty. Using first val patient as train.")
        train_list = [val_list[0]]
        val_list = val_list[1:]
    return train_list, val_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS 3-D U-Net trainer")
    p.add_argument("--config",   default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--run-name", default=None,
                   help="Override run directory name (creates runs/<run-name>/)")
    p.add_argument("--resume",   default=None,
                   help="Path to a last.pt checkpoint to resume from")
    p.add_argument("--subset",   default=None,
                   help="Path to a smaller data directory for quick smoke runs")
    p.add_argument("--epochs",   type=int, default=None,
                   help="Override epochs from config (useful for smoke runs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.subset:
        cfg["data_dir"] = args.subset
    if args.run_name:
        cfg["run_dir"] = f"runs/{args.run_name}"
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    data_dir = Path(cfg["data_dir"])
    run_dir  = Path(cfg["run_dir"])

    # ── Data directory guard ─────────────────────────────────────────────
    if not data_dir.exists():
        print(f"\nERROR: data directory not found: {data_dir}")
        print("\nTo download the BraTS 2020 dataset:")
        print("  1. pip install kaggle")
        print("  2. Place ~/.kaggle/kaggle.json with your API credentials")
        print("  3. kaggle datasets download awsaf49/brats20-dataset-training-validation")
        print("  4. Unzip so the layout is:")
        print("       data/raw/BraTS2020_TrainingData/BraTS20_Training_001/...")
        sys.exit(1)

    # ── Reproducibility ──────────────────────────────────────────────────
    seed_everything(cfg.get("seed", 42))

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: CUDA not available — training on CPU will be very slow.")

    # ── Datalists ────────────────────────────────────────────────────────
    val_frac = 1.0 - float(cfg.get("train_val_split", 0.8))
    seed     = int(cfg.get("seed", 42))

    train_list = get_brats_datalist(data_dir, split="train", val_frac=val_frac, seed=seed)
    val_list   = get_brats_datalist(data_dir, split="val",   val_frac=val_frac, seed=seed)
    train_list, val_list = _guard_splits(train_list, val_list)

    # Repeat train items to approximate samples_per_volume without changing transforms
    spv = int(cfg.get("samples_per_volume", 1))
    if spv > 1:
        train_list = [item for item in train_list for _ in range(spv)]

    print(f"Train patients: {len(train_list) // max(spv, 1)}  "
          f"(× {spv} = {len(train_list)} items)  |  Val patients: {len(val_list)}")

    # ── Transforms & datasets ────────────────────────────────────────────
    roi_size = tuple(cfg.get("patch_size", [96, 96, 96]))
    train_ds = make_dataset(train_list, get_train_transforms(roi_size), use_cache=False)
    val_ds   = make_dataset(val_list,   get_val_transforms(),           use_cache=False)

    num_workers = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,          # full volumes — keep batch size 1
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    # ── Model, loss, metric ──────────────────────────────────────────────
    model   = build_unet(cfg.get("model")).to(device)
    loss_fn = build_loss().to(device)
    metric  = RegionDice()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f} M")

    # ── W&B ─────────────────────────────────────────────────────────────
    if cfg.get("use_wandb", False):
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "brats-segmentation"),
            name=run_dir.name,
            config=cfg,
        )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metric=metric,
        config=cfg,
        device=device,
        run_dir=run_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print(f"\nStarting training → {run_dir}\n")
    trainer.fit()
    print("\nDone.")


if __name__ == "__main__":
    main()
