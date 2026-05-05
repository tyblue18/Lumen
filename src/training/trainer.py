"""Training loop for BraTS 3-D segmentation.

Responsibilities:
- AMP-wrapped forward/backward pass (torch.amp)
- AdamW + CosineAnnealingLR
- Sliding-window validation every val_frequency epochs
- Checkpoint saving: last.pt every epoch, best.pt on val-Dice improvement
- Epoch summaries appended to log.jsonl (one JSON object per line)
- Optional Weights & Biases logging (only if config["use_wandb"] is True)
"""

import json
import time
from pathlib import Path

import torch
import torch.amp
from monai.inferers import sliding_window_inference
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        metric,
        config: dict,
        device: torch.device,
        run_dir: Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metric = metric
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        n_epochs = config["epochs"]
        opt_cfg = config.get("optimizer", {})
        self.optimizer = AdamW(
            model.parameters(),
            lr=float(opt_cfg.get("lr", 1e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs)

        self.use_amp = config.get("amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.use_wandb = config.get("use_wandb", False)

        self.start_epoch = 1
        self.best_dice = -1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimiser, scheduler, scaler, and training state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_dice = ckpt.get("best_dice", -1.0)
        print(f"Resumed from epoch {ckpt['epoch']} (best Dice so far: {self.best_dice:.4f})")

    def fit(self) -> None:
        n_epochs = self.config["epochs"]
        val_freq = self.config.get("val_frequency", 1)
        on_gpu = self.device.type == "cuda"

        for epoch in range(self.start_epoch, n_epochs + 1):
            t0 = time.monotonic()

            if on_gpu:
                torch.cuda.reset_peak_memory_stats(self.device)
            train_loss = self._train_epoch(epoch, n_epochs)
            peak_train_mb = (
                torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
                if on_gpu else None
            )

            val_scores: dict | None = None
            peak_val_mb: float | None = None
            if epoch % val_freq == 0 and self.val_loader is not None:
                if on_gpu:
                    torch.cuda.reset_peak_memory_stats(self.device)
                val_scores = self._val_epoch(epoch, n_epochs)
                if on_gpu:
                    peak_val_mb = (
                        torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
                    )

            elapsed = time.monotonic() - t0

            is_best = False
            if val_scores is not None:
                mean_dice = sum(val_scores.values()) / len(val_scores)
                if mean_dice > self.best_dice:
                    self.best_dice = mean_dice
                    is_best = True

            self._save_checkpoint(epoch, is_best)
            self._log_epoch(epoch, train_loss, val_scores, elapsed,
                            peak_train_mb, peak_val_mb)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, n_epochs: int) -> float:
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{n_epochs} [train]",
            leave=True,
        )
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            running_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        return running_loss / max(len(self.train_loader), 1)

    def _val_epoch(self, epoch: int, n_epochs: int) -> dict:
        self.model.eval()
        roi_size = self.config.get("sliding_window_roi", [96, 96, 96])
        overlap = float(self.config.get("sliding_window_overlap", 0.5))

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc=f"Epoch {epoch}/{n_epochs} [val]",
                leave=True,
            ):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=2,
                    predictor=self.model,
                    overlap=overlap,
                )
                preds = (torch.sigmoid(outputs) > 0.5).float()
                self.metric.update(preds, labels)

        return self.metric.compute()  # also resets accumulator

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_dice": self.best_dice,
        }
        torch.save(state, self.run_dir / "last.pt")
        if is_best:
            torch.save(state, self.run_dir / "best.pt")
            print(f"  ✓ New best — saved best.pt  (mean Dice {self.best_dice:.4f})")

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_scores: dict | None,
        elapsed_s: float,
        peak_train_mb: float | None,
        peak_val_mb: float | None,
    ) -> None:
        entry: dict = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "lr": round(self.optimizer.param_groups[0]["lr"], 8),
            "time_s": round(elapsed_s, 1),
        }
        if peak_train_mb is not None:
            entry["peak_train_vram_mb"] = round(peak_train_mb, 0)
        if peak_val_mb is not None:
            entry["peak_val_vram_mb"] = round(peak_val_mb, 0)
        if val_scores is not None:
            for region, dice in val_scores.items():
                entry[f"val_{region}"] = round(float(dice), 6)
            entry["val_mean"] = round(
                sum(val_scores.values()) / len(val_scores), 6
            )
            entry["best_dice"] = round(self.best_dice, 6)

        log_path = self.run_dir / "log.jsonl"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        # Console summary
        parts = [f"epoch {epoch}  loss={train_loss:.4f}"]
        if peak_train_mb is not None:
            parts.append(f"VRAM={peak_train_mb/1024:.2f}GB")
        if val_scores:
            for k, v in val_scores.items():
                parts.append(f"{k}={float(v):.4f}")
        parts.append(f"time={elapsed_s:.0f}s")
        print("  " + "  ".join(parts))

        if self.use_wandb:
            import wandb
            wandb.log(entry)
