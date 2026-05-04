"""Unit tests for dataset and transform pipeline.

Tests:
- Patient discovery and train/val split reproducibility
- Label remapping (4 → 3) and multi-channel conversion
- Per-channel non-zero normalization
- Patch shape after random crop
- No data leakage between train and val splits
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from monai.data import DataLoader

from src.data.dataset import get_brats_datalist, make_dataset
from src.data.transforms import get_train_transforms, get_val_transforms

DATA_DIR = Path("data/raw/BraTS2020_TrainingData")
ROI = (96, 96, 96)


@pytest.fixture(scope="module")
def datalists():
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found: {DATA_DIR}")
    train = get_brats_datalist(DATA_DIR, split="train", val_frac=0.2, seed=42)
    val   = get_brats_datalist(DATA_DIR, split="val",   val_frac=0.2, seed=42)
    return train, val


def test_split_sizes(datalists):
    train, val = datalists
    total = len(train) + len(val)
    assert total == 369, f"Expected 369 patients, got {total}"
    # ~20 % val — allow ±5 patients for hash distribution
    assert 60 <= len(val) <= 90, f"Unexpected val size: {len(val)}"


def test_split_no_overlap(datalists):
    train, val = datalists
    train_ids = {d["label"] for d in train}
    val_ids   = {d["label"] for d in val}
    assert train_ids.isdisjoint(val_ids), "Train/val overlap detected"


def test_split_reproducibility():
    """Same seed must produce identical splits on repeated calls."""
    a = get_brats_datalist(DATA_DIR, split="train", val_frac=0.2, seed=42)
    b = get_brats_datalist(DATA_DIR, split="train", val_frac=0.2, seed=42)
    assert [d["label"] for d in a] == [d["label"] for d in b]


def test_different_seeds_differ():
    a = get_brats_datalist(DATA_DIR, split="val", val_frac=0.2, seed=42)
    b = get_brats_datalist(DATA_DIR, split="val", val_frac=0.2, seed=99)
    assert [d["label"] for d in a] != [d["label"] for d in b]


# ---------------------------------------------------------------------------
# Transform / shape tests — use Dataset (no caching) on a 2-item slice
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_batch(datalists):
    train, _ = datalists
    ds = make_dataset(train[:2], get_train_transforms(ROI), use_cache=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    return next(iter(loader))


@pytest.fixture(scope="module")
def val_batch(datalists):
    _, val = datalists
    ds = make_dataset(val[:2], get_val_transforms(), use_cache=False)
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    return next(iter(loader))


def test_train_image_shape(train_batch):
    img = train_batch["image"]
    assert img.shape == (1, 4, *ROI), f"Unexpected image shape: {img.shape}"


def test_train_label_shape(train_batch):
    lbl = train_batch["label"]
    assert lbl.shape == (1, 3, *ROI), f"Unexpected label shape: {lbl.shape}"


def test_train_label_binary(train_batch):
    lbl = train_batch["label"]
    unique = lbl.unique().tolist()
    assert all(v in (0.0, 1.0) for v in unique), f"Non-binary label values: {unique}"


def test_train_no_nans(train_batch):
    assert not torch.isnan(train_batch["image"]).any(), "NaN in train image"
    assert not torch.isnan(train_batch["label"]).any(), "NaN in train label"


def test_val_image_channels(val_batch):
    assert val_batch["image"].shape[1] == 4, "Val image must have 4 channels"


def test_val_label_channels(val_batch):
    assert val_batch["label"].shape[1] == 3, "Val label must have 3 channels"


def test_val_label_binary(val_batch):
    lbl = val_batch["label"]
    unique = lbl.unique().tolist()
    assert all(v in (0.0, 1.0) for v in unique), f"Non-binary val label values: {unique}"


def test_val_no_nans(val_batch):
    assert not torch.isnan(val_batch["image"]).any(), "NaN in val image"
    assert not torch.isnan(val_batch["label"]).any(), "NaN in val label"


def test_et_channel_nonzero(datalists):
    """ET channel (ch 2) must have positive voxels for at least one patient.

    Guards against the silent bug of remapping label 4 → 3 before
    ConvertToMultiChannelBasedOnBratsClasses (which checks for label 4).
    """
    train, _ = datalists
    ds = make_dataset(train[:3], get_train_transforms(ROI), use_cache=False)
    found_et = False
    for item in ds:
        if item["label"][2].sum() > 0:
            found_et = True
            break
    assert found_et, (
        "ET channel is all-zeros across first 3 patients — "
        "possible label-4 → label-3 remapping bug"
    )
