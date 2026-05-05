"""Evaluate a checkpoint on the BraTS 2020 validation set.

Usage:
    python scripts/evaluate.py --ckpt runs/brats_v1/best.pt
    python scripts/evaluate.py --ckpt runs/brats_v1/best.pt --hd95
    python scripts/evaluate.py --ckpt runs/brats_v1/best.pt --output-dir runs/brats_v1/eval/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import get_brats_datalist, make_dataset
from src.data.transforms import get_val_transforms
from src.models.unet import build_unet
from src.utils.visualize import build_qualitative_grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a BraTS checkpoint")
    p.add_argument("--ckpt",       required=True,
                   help="Path to .pt checkpoint (e.g. runs/brats_v1/best.pt)")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--output-dir", default=None,
                   help="Where to write eval.json / qualitative.png "
                        "(default: same directory as --ckpt)")
    p.add_argument("--seed",       type=int, default=42,
                   help="RNG seed for qualitative patient selection")
    p.add_argument("--hd95",       action="store_true",
                   help="Also compute HD95 (slow — adds ~30 s per patient on CPU)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md_table(rows: list[tuple], header: list[str]) -> str:
    col_w = [max(len(h), max(len(str(r[i])) for r in rows))
             for i, h in enumerate(header)]
    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    return "\n".join([fmt_row(header), sep] + [fmt_row(r) for r in rows])


def _nanstat(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) ignoring NaN entries."""
    arr = np.array(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ckpt_path  = Path(args.ckpt)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ──────────────────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_dir   = Path(cfg["data_dir"])
    val_frac   = 1.0 - float(cfg.get("train_val_split", 0.8))
    seed       = int(cfg.get("seed", 42))
    roi_size   = cfg.get("sliding_window_roi", [96, 96, 96])
    overlap    = float(cfg.get("sliding_window_overlap", 0.5))

    if not data_dir.exists():
        sys.exit(f"ERROR: data_dir not found: {data_dir}")

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Model ───────────────────────────────────────────────────────────
    model = build_unet(cfg.get("model")).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    trained_epoch = ckpt.get("epoch", "?")
    print(f"Loaded : {ckpt_path}  (epoch {trained_epoch}, "
          f"stored best_dice={ckpt.get('best_dice', '?'):.4f})")

    # ── Val DataLoader ───────────────────────────────────────────────────
    val_list = get_brats_datalist(data_dir, split="val",
                                  val_frac=val_frac, seed=seed)
    if not val_list:
        sys.exit("ERROR: val split is empty — check data_dir / seed / val_frac")

    val_ds = make_dataset(val_list, get_val_transforms(), use_cache=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=int(cfg.get("num_workers", 0)),
                            pin_memory=device.type == "cuda")
    print(f"Val set: {len(val_list)} patients\n")

    # ── Qualitative patient selection ─────────────────────────────────
    rng = np.random.default_rng(args.seed)
    qual_n = min(4, len(val_list))
    qual_indices = set(
        int(i) for i in rng.choice(len(val_list), size=qual_n, replace=False)
    )
    qual_rows: dict[int, dict] = {}

    # ── Metrics ─────────────────────────────────────────────────────────
    # reduction="none" buffers per-sample, per-channel scores for std computation
    dice_metric = DiceMetric(include_background=True, reduction="none",
                             get_not_nans=False)
    hd_metric   = (HausdorffDistanceMetric(include_background=True,
                                           percentile=95, reduction="none",
                                           directed=False, get_not_nans=False)
                   if args.hd95 else None)

    if args.hd95:
        print("WARNING: --hd95 enabled. Expect ~30–60 s extra per patient on CPU.\n")

    # ── Inference loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Sliding-window inference")):
            images = batch["image"].to(device)   # (1, 4, H, W, D)
            labels = batch["label"].to(device)   # (1, 3, H, W, D)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=2,
                predictor=model,
                overlap=overlap,
            )
            preds = (torch.sigmoid(logits) > 0.5).float()

            dice_metric(y_pred=preds, y=labels)
            if hd_metric is not None:
                hd_metric(y_pred=preds, y=labels)

            # Capture mid-axial slice for qualitative grid
            if idx in qual_indices:
                d = images.shape[-1] // 2
                qual_rows[idx] = {
                    "patient_id": Path(val_list[idx]["label"]).parent.name,
                    # Transpose to (W, H) so imshow origin="lower" gives
                    # the conventional axial orientation (A at top)
                    "flair": images[0, 3, :, :, d].cpu().numpy(),
                    "gt"   : labels[0, :, :, :, d].cpu().numpy(),
                    "pred" : preds [0, :, :, :, d].cpu().numpy(),
                }

    # ── Aggregate Dice ───────────────────────────────────────────────────
    # Shape: (N_patients, 3) — columns are [TC, WT, ET]
    all_dices = dice_metric.aggregate().cpu().numpy()
    dice_metric.reset()
    # Map channel indices to region names (same order as in metrics.py)
    ch_names = ["TC", "WT", "ET"]

    per_patient: list[dict] = []
    for i, item in enumerate(val_list):
        pid = Path(item["label"]).parent.name
        entry: dict = {"patient_id": pid}
        for j, name in enumerate(ch_names):
            v = float(all_dices[i, j])
            entry[f"dice_{name}"] = None if np.isnan(v) else round(v, 6)
        per_patient.append(entry)

    # HD95 per-patient
    if hd_metric is not None:
        all_hd = hd_metric.aggregate().cpu().numpy()
        hd_metric.reset()
        for i, entry in enumerate(per_patient):
            for j, name in enumerate(ch_names):
                v = float(all_hd[i, j])
                entry[f"hd95_{name}"] = None if (np.isnan(v) or np.isinf(v)) else round(v, 4)

    # ── Aggregate statistics ─────────────────────────────────────────────
    agg: dict = {}
    for j, name in enumerate(ch_names):
        vals = [float(all_dices[i, j]) for i in range(len(val_list))]
        mean, std = _nanstat(vals)
        agg[f"dice_{name}"] = {"mean": round(mean, 6), "std": round(std, 6)}

    dice_means = [agg[f"dice_{n}"]["mean"] for n in ch_names]
    dice_stds  = [agg[f"dice_{n}"]["std"]  for n in ch_names]
    agg["dice_mean"] = {
        "mean": round(float(np.mean(dice_means)), 6),
        "std" : round(float(np.mean(dice_stds)),  6),
    }

    if hd_metric is not None:
        for j, name in enumerate(ch_names):
            vals = [float(all_hd[i, j]) for i in range(len(val_list))]
            # filter inf before nanstat
            vals = [v for v in vals if not np.isinf(v)]
            mean, std = _nanstat(vals) if vals else (float("nan"), float("nan"))
            agg[f"hd95_{name}"] = {"mean": round(mean, 4), "std": round(std, 4)}

    # ── Markdown table ───────────────────────────────────────────────────
    header = ["Region", "Mean Dice", "Std Dice"]
    if hd_metric is not None:
        header += ["Mean HD95 (mm)", "Std HD95 (mm)"]

    table_rows = []
    for name in ["WT", "TC", "ET"]:  # clinical reporting order: WT, TC, ET
        d_agg = agg[f"dice_{name}"]
        row = [name, f"{d_agg['mean']:.4f}", f"{d_agg['std']:.4f}"]
        if hd_metric is not None:
            h_agg = agg.get(f"hd95_{name}", {"mean": float("nan"), "std": float("nan")})
            row += [f"{h_agg['mean']:.2f}", f"{h_agg['std']:.2f}"]
        table_rows.append(tuple(row))

    # Mean row
    m_agg = agg["dice_mean"]
    mean_row = ["**Mean**", f"**{m_agg['mean']:.4f}**", f"**{m_agg['std']:.4f}**"]
    if hd_metric is not None:
        mean_row += ["—", "—"]
    table_rows.append(tuple(mean_row))

    print("\n## Evaluation results\n")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Epoch      : {trained_epoch}")
    print(f"Val set    : {len(val_list)} patients\n")
    print(_md_table(table_rows, header))
    print()

    # ── Save eval.json ───────────────────────────────────────────────────
    eval_json = {
        "checkpoint"   : str(ckpt_path),
        "epoch"        : trained_epoch,
        "config"       : str(args.config),
        "val_patients" : len(val_list),
        "aggregate"    : agg,
        "per_patient"  : per_patient,
    }
    json_path = output_dir / "eval.json"
    json_path.write_text(json.dumps(eval_json, indent=2), encoding="utf-8")
    print(f"Saved eval.json → {json_path}")

    # ── Qualitative grid ─────────────────────────────────────────────────
    sorted_rows = [qual_rows[i] for i in sorted(qual_rows)]
    grid_path = build_qualitative_grid(sorted_rows, output_dir / "qualitative.png")
    print(f"Saved qualitative.png → {grid_path}")


if __name__ == "__main__":
    main()
