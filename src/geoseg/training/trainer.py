"""Training / validation loop for the change detection baseline."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from geoseg.data.dataset import ChangeDetectionDataset
from geoseg.models.siamese_unet import SiameseUNet
from geoseg.training.config import Config
from geoseg.training.losses import build_loss
from geoseg.training.metrics import BinaryConfusion
from geoseg.training.tracking import flatten, make_tracker

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: Config) -> SiameseUNet:
    return SiameseUNet(
        encoder_name=cfg.model.encoder,
        encoder_weights=cfg.model.encoder_weights,
        fusion=cfg.model.fusion,
    )


def build_dataloaders(
    cfg: Config, pin_memory: bool = False
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Train loader (shuffled, augmented) and val loader (ordered, no augmentation)."""
    data, train = cfg.data, cfg.train
    train_ds = ChangeDetectionDataset(
        data.root, "train", data.tile_size, data.overlap, augment=True, cache=data.cache
    )
    val_ds = ChangeDetectionDataset(
        data.root, "val", data.tile_size, 0, augment=False, cache=data.cache
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    common: dict[str, Any] = {
        "batch_size": train.batch_size,
        "num_workers": data.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": data.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    log_every: int = 0,
    limit_batches: int = 0,
    epoch: int = 0,
) -> float:
    """One pass over ``loader``; returns the mean training loss."""
    model.train()
    total, count = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        image_a = batch["image_a"].to(device, non_blocking=True)
        image_b = batch["image_b"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(image_a, image_b)
        loss = loss_fn(logits.float(), mask)  # loss in float32 even under AMP
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = image_a.shape[0]
        total += loss.item() * batch_size
        count += batch_size
        if log_every and step % log_every == 0:
            logger.info("epoch %d step %d/%d loss %.4f", epoch, step, len(loader), total / count)
        if limit_batches and step >= limit_batches:
            break
    return total / max(count, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    loss_fn: nn.Module,
    device: torch.device,
    *,
    amp: bool,
    threshold: float = 0.5,
    limit_batches: int = 0,
) -> dict[str, float]:
    """Mean loss + IoU/F1/precision/recall of the change class over ``loader``."""
    model.eval()
    confusion = BinaryConfusion()
    total, count = 0.0, 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            image_a = batch["image_a"].to(device, non_blocking=True)
            image_b = batch["image_b"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(image_a, image_b)
            logits = logits.float()
            total += loss_fn(logits, mask).item() * image_a.shape[0]
            count += image_a.shape[0]
            confusion.update(logits, mask, threshold)
            if limit_batches and step >= limit_batches:
                break
    metrics = confusion.compute()
    metrics["loss"] = total / max(count, 1)
    return metrics


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: Config,
    epoch: int,
    val_metrics: dict[str, float],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_f1: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> None:
    """Save a full checkpoint: model weights plus everything needed to resume training.

    ``model``/``config``/``epoch``/``val`` are also used by callers that only want the
    trained weights (e.g. inference); the rest is resume-only bookkeeping.
    """
    payload = {
        "model": model.state_dict(),
        "config": cfg.model_dump(),
        "epoch": epoch,
        "val": val_metrics,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
    }
    torch.save(payload, path)


def _resolve_resume_path(resume: str | Path | None, output_dir: Path) -> Path | None:
    """``None``/``"none"`` -> no resume. ``"auto"`` -> ``output_dir/last.pt`` if it exists.

    Anything else is treated as an explicit checkpoint path (must exist).
    """
    if resume is None or resume == "none":
        return None
    if resume == "auto":
        candidate = output_dir / "last.pt"
        return candidate if candidate.is_file() else None
    path = Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    return path


def _load_resume_state(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    output_dir: Path,
) -> tuple[int, float, int, int, list[dict[str, float]]]:
    """Restore model/optimizer/scheduler/scaler from ``path``.

    Returns ``(start_epoch, best_f1, best_epoch, epochs_without_improvement, history)``.
    Checkpoints saved before resume support lack the extra keys; those are filled with
    safe defaults and a warning is logged (only the model weights carry over cleanly).
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    missing = [k for k in ("optimizer", "scheduler", "scaler") if k not in checkpoint]
    if missing:
        logger.warning(
            "Checkpoint %s has no %s (older format): resuming weights only, "
            "optimizer/scheduler state restarts fresh.",
            path,
            ", ".join(missing),
        )
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])

    history_path = output_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else []

    start_epoch = int(checkpoint["epoch"]) + 1
    best_f1 = float(checkpoint.get("best_f1", checkpoint.get("val", {}).get("f1", -1.0)))
    best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    logger.info("Resuming from %s: starting at epoch %d", path, start_epoch)
    return start_epoch, best_f1, best_epoch, epochs_without_improvement, history


def fit(cfg: Config, resume: str | Path | None = None) -> dict[str, Any]:
    """Train according to ``cfg``; returns a summary of the best epoch (by val F1).

    Writes to ``cfg.logging.output_dir``: ``config.yaml``, ``history.json`` (updated every
    epoch), ``last.pt`` and ``best.pt`` (best validation F1).

    ``resume``: ``None``/``"none"`` starts fresh (default). ``"auto"`` continues from
    ``output_dir/last.pt`` if it exists, else starts fresh. Any other value is treated as
    an explicit checkpoint path. Resuming restores model, optimizer, scheduler, AMP
    scaler, early-stopping counters and prior history, then continues from the next epoch.
    """
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.train.amp and device.type == "cuda"
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8"
    )

    train_loader, val_loader = build_dataloaders(cfg, pin_memory=device.type == "cuda")
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg.train.loss, cfg.train.focal_gamma, cfg.train.focal_alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "device=%s amp=%s params=%.1fM train_tiles=%d val_tiles=%d",
        device,
        use_amp,
        n_params / 1e6,
        len(train_loader.dataset),  # type: ignore[arg-type]
        len(val_loader.dataset),  # type: ignore[arg-type]
    )

    resume_path = _resolve_resume_path(resume, output_dir)
    history: list[dict[str, float]] = []
    best_f1, best_epoch, best_val = -1.0, 0, {}
    epochs_without_improvement = 0
    start_epoch = 1
    if resume_path is not None:
        start_epoch, best_f1, best_epoch, epochs_without_improvement, history = _load_resume_state(
            resume_path, model, optimizer, scheduler, scaler, device, output_dir
        )
        if history:
            best_val = next(
                (
                    {k[len("val_") :]: v for k, v in row.items() if k.startswith("val_")}
                    for row in history
                    if int(row["epoch"]) == best_epoch
                ),
                {},
            )

    tracker = make_tracker(cfg, output_dir)
    tracker.log_params(flatten(cfg.model_dump()))

    if start_epoch > cfg.train.epochs:
        logger.info(
            "Resume epoch %d already >= configured epochs %d; nothing to do.",
            start_epoch,
            cfg.train.epochs,
        )

    try:
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            start = time.perf_counter()
            train_loss = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                scaler,
                device,
                amp=use_amp,
                log_every=cfg.train.log_every,
                limit_batches=cfg.train.limit_batches,
                epoch=epoch,
            )
            val = evaluate(
                model,
                val_loader,
                loss_fn,
                device,
                amp=use_amp,
                threshold=cfg.train.threshold,
                limit_batches=cfg.train.limit_batches,
            )
            row = {
                "epoch": float(epoch),
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val.items()},
                "seconds": time.perf_counter() - start,
            }
            scheduler.step()
            history.append(row)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2))
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            logger.info(
                "epoch %d/%d train_loss %.4f val_loss %.4f val_iou %.4f val_f1 %.4f (%.0fs)",
                epoch,
                cfg.train.epochs,
                train_loss,
                val["loss"],
                val["iou"],
                val["f1"],
                row["seconds"],
            )

            improved = val["f1"] > best_f1
            if improved:
                best_f1, best_epoch, best_val = val["f1"], epoch, val
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            checkpoint_kwargs: dict[str, Any] = {
                "optimizer": optimizer,
                "scheduler": scheduler,
                "scaler": scaler,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            }
            # Bookkeeping (best_f1/epochs_without_improvement) is updated above BEFORE
            # saving, so last.pt always reflects the state to resume from correctly.
            _save_checkpoint(output_dir / "last.pt", model, cfg, epoch, val, **checkpoint_kwargs)
            if improved:
                _save_checkpoint(
                    output_dir / "best.pt", model, cfg, epoch, val, **checkpoint_kwargs
                )
            elif cfg.train.patience and epochs_without_improvement >= cfg.train.patience:
                logger.info(
                    "Early stopping: no val F1 improvement for %d epochs", epoch - best_epoch
                )
                break
        tracker.log_artifact(output_dir / "config.yaml")
        tracker.log_artifact(output_dir / "history.json")
    finally:
        tracker.close()

    return {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "epochs_run": len(history),
        "output_dir": str(output_dir),
    }
