"""Shared fixtures. Only NumPy/PIL here, so they work without torch (as in CI)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def write_change_split(root: Path, split: str, n_images: int, size: int, seed: int) -> None:
    """Synthetic LEVIR-CD-like split: image B = image A + a bright square = the change."""
    rng = np.random.default_rng(seed)
    for sub in ("A", "B", "label"):
        (root / split / sub).mkdir(parents=True)
    for i in range(n_images):
        name = f"img_{i}.png"
        before = rng.integers(0, 150, (size, size, 3), dtype=np.uint8)
        after = before.copy()
        y, x = rng.integers(8, size - 40, size=2)
        after[y : y + 32, x : x + 32] = 255
        label = np.zeros((size, size), dtype=np.uint8)
        label[y : y + 32, x : x + 32] = 255
        Image.fromarray(before).save(root / split / "A" / name)
        Image.fromarray(after).save(root / split / "B" / name)
        Image.fromarray(label).save(root / split / "label" / name)


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """128x128 images, 3 train + 2 val; with tile_size=64 that is 12 train / 8 val tiles."""
    write_change_split(tmp_path, "train", n_images=3, size=128, seed=0)
    write_change_split(tmp_path, "val", n_images=2, size=128, seed=1)
    return tmp_path
