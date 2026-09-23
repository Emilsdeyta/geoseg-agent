"""Evaluate a trained checkpoint on a held-out split (default: test).

Reuses the exact same tile-level IoU/F1 computation used during training/validation
(``geoseg.training.trainer.evaluate``), so these numbers are directly comparable to the
``val_*`` numbers in ``history.json``.

Usage::

    python -m geoseg.evaluation.evaluate --checkpoint outputs/full_run/best.pt \\
        --data-root /path/to/levir-cd

    # different split, save metrics elsewhere, more dataloader workers
    python -m geoseg.evaluation.evaluate --checkpoint outputs/full_run/best.pt \\
        --data-root /path/to/levir-cd --split val --output outputs/full_run/val_metrics.json

The checkpoint carries the config it was trained with (model architecture, tile size,
loss). ``--data-root`` is the one thing you almost always need to override, since a
checkpoint made on Kaggle stores the Kaggle data path.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from geoseg.data.dataset import ChangeDetectionDataset
from geoseg.training.config import Config
from geoseg.training.losses import build_loss
from geoseg.training.trainer import build_model, evaluate

logger = logging.getLogger(__name__)


def load_checkpoint_config(checkpoint: dict, data_root: str | None) -> Config:
    cfg = Config.model_validate(checkpoint["config"])
    if data_root is not None:
        cfg.data.root = data_root
    return cfg


def run_evaluation(
    checkpoint_path: Path,
    *,
    data_root: str | None = None,
    split: str = "test",
    batch_size: int | None = None,
    num_workers: int | None = None,
    cache: bool = False,
    threshold: float | None = None,
) -> dict[str, float | int | str]:
    """Load ``checkpoint_path`` and evaluate it on ``split``. Returns a metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = load_checkpoint_config(checkpoint, data_root)

    dataset = ChangeDetectionDataset(
        cfg.data.root,
        split,
        cfg.data.tile_size,
        overlap=0,  # non-overlapping tiles, same as validation
        augment=False,
        cache=cache,
    )
    loader: DataLoader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    n_images = len({name for name, _, _ in dataset.index})
    logger.info(
        "split=%s images=%d tiles=%d tile_size=%d",
        split,
        n_images,
        len(dataset),
        cfg.data.tile_size,
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    loss_fn = build_loss(cfg.train.loss, cfg.train.focal_gamma, cfg.train.focal_alpha)

    metrics = evaluate(
        model,
        loader,
        loss_fn,
        device,
        amp=False,  # eval-only, no need for AMP; keeps numbers deterministic
        threshold=threshold if threshold is not None else cfg.train.threshold,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": split,
        "n_images": n_images,
        "n_tiles": len(dataset),
        **metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a data split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the dataset root stored in the checkpoint's config "
        "(usually needed: a Kaggle-trained checkpoint stores the Kaggle path).",
    )
    parser.add_argument("--split", default="test", help="train | val | test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache", action="store_true", help="Cache all images in RAM.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Where to save metrics as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    metrics = run_evaluation(
        args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache=args.cache,
        threshold=args.threshold,
    )
    print(json.dumps(metrics, indent=2))

    output = args.output or args.checkpoint.parent / f"{args.split}_metrics.json"
    output.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
