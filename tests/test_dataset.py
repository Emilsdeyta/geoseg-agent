from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from geoseg.data.dataset import ChangeDetectionDataset  # noqa: E402


def _make_split(root: Path, split: str, n_images: int = 2, size: int = 300) -> None:
    rng = np.random.default_rng(0)
    for sub in ("A", "B", "label"):
        (root / split / sub).mkdir(parents=True)
    for i in range(n_images):
        name = f"img_{i}.png"
        for sub in ("A", "B"):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / split / sub / name)
        label = np.zeros((size, size), dtype=np.uint8)
        label[50:120, 50:120] = 255
        Image.fromarray(label).save(root / split / "label" / name)


def test_length_and_shapes(tmp_path: Path) -> None:
    _make_split(tmp_path, "train")
    ds = ChangeDetectionDataset(tmp_path, "train", tile_size=256)
    assert len(ds) == 2 * 4  # 300px image -> 2x2 tiles
    sample = ds[0]
    assert sample["image_a"].shape == (3, 256, 256)
    assert sample["image_b"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}


def test_augmentation_keeps_shapes_and_mask_values(tmp_path: Path) -> None:
    _make_split(tmp_path, "train")
    ds = ChangeDetectionDataset(tmp_path, "train", tile_size=128, augment=True)
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["mask"].shape == (1, 128, 128)
        assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ChangeDetectionDataset(tmp_path, "train")


def test_cache_gives_identical_samples(tmp_path: Path) -> None:
    _make_split(tmp_path, "train")
    plain = ChangeDetectionDataset(tmp_path, "train", tile_size=128)
    cached = ChangeDetectionDataset(tmp_path, "train", tile_size=128, cache=True)
    assert len(plain) == len(cached)
    for i in range(len(plain)):
        for key in ("image_a", "image_b", "mask"):
            assert torch.equal(plain[i][key], cached[i][key])


def test_augmentation_does_not_corrupt_the_cache(tmp_path: Path) -> None:
    """Tiles are views of the cached arrays; augmenting must never write into them."""
    _make_split(tmp_path, "train")
    cached = ChangeDetectionDataset(tmp_path, "train", tile_size=128, augment=True, cache=True)
    for _ in range(3):
        for i in range(len(cached)):
            cached[i]
    cached.augment = False
    plain = ChangeDetectionDataset(tmp_path, "train", tile_size=128)
    for i in range(len(plain)):
        for key in ("image_a", "image_b", "mask"):
            assert torch.equal(plain[i][key], cached[i][key])
