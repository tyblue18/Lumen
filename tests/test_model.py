"""Tests for model construction, forward pass, loss, metrics, and gradient flow."""

import torch
import pytest
from src.models.unet import build_unet
from src.training.losses import build_loss
from src.training.metrics import RegionDice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, C_IN, C_OUT = 1, 4, 3
PATCH = (96, 96, 96)


@pytest.fixture(scope="module")
def model():
    return build_unet().to(DEVICE)


@pytest.fixture(scope="module")
def random_batch():
    x = torch.randn(B, C_IN, *PATCH, device=DEVICE)
    # Binary multi-channel label matching ConvertToMultiChannelBasedOnBratsClassesd output
    y = torch.randint(0, 2, (B, C_OUT, *PATCH), dtype=torch.float32, device=DEVICE)
    return x, y


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def test_output_shape(model, random_batch):
    x, _ = random_batch
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, C_OUT, *PATCH), f"Unexpected output shape: {out.shape}"


def test_output_is_logits(model, random_batch):
    """Model must return raw logits — values should exceed [0, 1]."""
    x, _ = random_batch
    with torch.no_grad():
        out = model(x)
    # At least some logits should be outside [0, 1] for an untrained model
    assert (out.abs() > 1.0).any(), "Output looks like probabilities, not logits"


def test_no_output_nans(model, random_batch):
    x, _ = random_batch
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "NaN in model output"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_loss_finite(model, random_batch):
    loss_fn = build_loss().to(DEVICE)
    x, y = random_batch
    with torch.no_grad():
        out = model(x)
    loss = loss_fn(out, y)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_loss_positive(model, random_batch):
    loss_fn = build_loss().to(DEVICE)
    x, y = random_batch
    with torch.no_grad():
        out = model(x)
    loss = loss_fn(out, y)
    assert loss.item() > 0, "Loss should be positive for random predictions"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow():
    """All parameters must receive gradients after a backward pass."""
    m = build_unet().to(DEVICE)
    loss_fn = build_loss().to(DEVICE)
    x = torch.randn(B, C_IN, *PATCH, device=DEVICE)
    y = torch.randint(0, 2, (B, C_OUT, *PATCH), dtype=torch.float32, device=DEVICE)

    loss = loss_fn(m(x), y)
    loss.backward()

    no_grad = [
        name for name, p in m.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not no_grad, f"Parameters without gradients: {no_grad[:5]}"


def test_gradients_nonzero():
    """At least some gradients must be non-zero (guards against dead network)."""
    m = build_unet().to(DEVICE)
    loss_fn = build_loss().to(DEVICE)
    x = torch.randn(B, C_IN, *PATCH, device=DEVICE)
    y = torch.randint(0, 2, (B, C_OUT, *PATCH), dtype=torch.float32, device=DEVICE)

    loss_fn(m(x), y).backward()

    all_zero = all(
        p.grad.abs().max().item() == 0.0
        for p in m.parameters() if p.grad is not None
    )
    assert not all_zero, "All gradients are zero"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_region_dice_keys():
    metric = RegionDice()
    preds   = torch.ones(B, C_OUT, 8, 8, 8)
    targets = torch.ones(B, C_OUT, 8, 8, 8)
    metric.update(preds, targets)
    scores = metric.compute()
    assert set(scores.keys()) == {"TC", "WT", "ET"}, f"Unexpected keys: {scores.keys()}"


def test_region_dice_perfect():
    """Perfect predictions → Dice = 1.0 for all regions."""
    metric = RegionDice()
    preds   = torch.ones(B, C_OUT, 8, 8, 8)
    targets = torch.ones(B, C_OUT, 8, 8, 8)
    metric.update(preds, targets)
    scores = metric.compute()
    for region, val in scores.items():
        assert abs(val - 1.0) < 1e-5, f"{region} Dice should be 1.0, got {val}"


def test_region_dice_channel_order():
    """Guard against TC/WT/ET mapping being transposed.

    Construct a label where only channel 0 (TC) is all-ones; channels 1 and 2
    are all-zeros.  Perfect predictions for TC should give Dice=1 only for TC.
    """
    metric = RegionDice()
    targets = torch.zeros(B, C_OUT, 8, 8, 8)
    targets[:, 0] = 1.0   # only TC foreground
    preds = torch.zeros(B, C_OUT, 8, 8, 8)
    preds[:, 0] = 1.0     # predict TC correctly
    metric.update(preds, targets)
    scores = metric.compute()
    assert abs(scores["TC"] - 1.0) < 1e-5, f"TC Dice should be 1.0, got {scores['TC']}"
    assert scores["WT"] < 0.1, f"WT Dice should be near 0, got {scores['WT']}"
    assert scores["ET"] < 0.1, f"ET Dice should be near 0, got {scores['ET']}"


def test_region_dice_resets_between_calls():
    metric = RegionDice()
    ones = torch.ones(B, C_OUT, 8, 8, 8)
    metric.update(ones, ones)
    metric.compute()  # resets internally
    # Second call with zeros/ones should give 0 Dice, not residual from first call
    zeros   = torch.zeros(B, C_OUT, 8, 8, 8)
    metric.update(zeros, ones)
    scores = metric.compute()
    for region, val in scores.items():
        assert val < 0.1, f"{region} Dice should be ~0 after reset, got {val}"
