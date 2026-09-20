"""Sanity-check a change detection dataset download.

Prints split sizes, tile counts and the fraction of changed pixels, and saves one
example tile (before | after | mask) as a PNG.

Usage:
    python scripts/check_data.py --root data/raw/levir-cd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.data.dataset import ChangeDetectionDataset


def change_ratio(dataset: ChangeDetectionDataset) -> float:
    """Fraction of pixels labelled as change across all masks of a split."""
    changed = 0
    total = 0
    for path in sorted(dataset.dir_label.iterdir()):
        with Image.open(path) as img:
            mask = np.asarray(img.convert("L")) > 127
        changed += int(mask.sum())
        total += mask.size
    return changed / total


def save_example(dataset: ChangeDetectionDataset, out_path: Path) -> None:
    """Save the first tile that contains a noticeable amount of change."""
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = np.asarray(sample["mask"])[0]
        if mask.mean() > 0.02:
            break
    before = (np.asarray(sample["image_a"]).transpose(1, 2, 0) * 255).astype(np.uint8)
    after = (np.asarray(sample["image_b"]).transpose(1, 2, 0) * 255).astype(np.uint8)
    mask_rgb = np.repeat((mask[..., None] * 255).astype(np.uint8), 3, axis=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([before, after, mask_rgb], axis=1)).save(out_path)
    print(f"Example saved: {out_path} ({sample['name']}, y={sample['y']}, x={sample['x']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/levir-cd")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--out", default="outputs/sample_tile.png")
    args = parser.parse_args()

    train_ds = None
    for split in ("train", "val", "test"):
        ds = ChangeDetectionDataset(args.root, split, tile_size=args.tile_size, normalize=False)
        n_images = len({name for name, _, _ in ds.index})
        with Image.open(ds.dir_a / ds.index[0][0]) as img:
            size = img.size
        print(
            f"{split:5s}: {n_images} image pairs, image size {size[0]}x{size[1]}, "
            f"{len(ds)} tiles of {args.tile_size}px, "
            f"changed pixels: {change_ratio(ds) * 100:.2f}%"
        )
        if split == "train":
            train_ds = ds

    if train_ds is not None:
        save_example(train_ds, Path(args.out))


if __name__ == "__main__":
    main()
