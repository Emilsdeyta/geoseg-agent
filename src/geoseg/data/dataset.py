"""PyTorch dataset for bi-temporal change detection (LEVIR-CD folder layout).

Expected layout::

    root/
      train/  A/  B/  label/
      val/    A/  B/  label/
      test/   A/  B/  label/

A = image before, B = image after, label = change mask (0 / 255).
Files with the same name in A, B and label belong together.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from geoseg.data.tiling import compute_tile_grid, extract_tile

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ChangeDetectionDataset(Dataset[dict[str, Any]]):
    """Serves fixed-size tiles of (image_a, image_b, change mask) triplets."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile_size: int = 256,
        overlap: int = 0,
        augment: bool = False,
        normalize: bool = True,
        cache: bool = False,
    ) -> None:
        # cache=True decodes every image once and keeps it in RAM (uint8). LEVIR-CD train is
        # ~3 GB. Without it each tile re-decodes three 1024x1024 PNGs, which starves the GPU.
        # With num_workers > 0 use it on Linux (fork shares the memory); on Windows every
        # worker would get its own copy.
        self.tile_size = tile_size
        self.augment = augment
        self.normalize = normalize

        split_dir = Path(root) / split
        self.dir_a = split_dir / "A"
        self.dir_b = split_dir / "B"
        self.dir_label = split_dir / "label"
        for directory in (self.dir_a, self.dir_b, self.dir_label):
            if not directory.is_dir():
                raise FileNotFoundError(f"Directory not found: {directory}")

        names = sorted(p.name for p in self.dir_a.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not names:
            raise FileNotFoundError(f"No images found in {self.dir_a}")

        self.index: list[tuple[str, int, int]] = []
        for name in names:
            for directory in (self.dir_b, self.dir_label):
                if not (directory / name).exists():
                    raise FileNotFoundError(f"Missing pair file: {directory / name}")
            with Image.open(self.dir_a / name) as img:
                width, height = img.size
            for y, x in compute_tile_grid(height, width, tile_size, overlap):
                self.index.append((name, y, x))

        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        if cache:
            self._cache = self._build_cache(names)

    def _build_cache(
        self, names: list[str]
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        start = time.perf_counter()
        cache = {
            name: (
                self._load(self.dir_a / name, "RGB"),
                self._load(self.dir_b / name, "RGB"),
                self._load(self.dir_label / name, "L"),
            )
            for name in names
        }
        size_gb = sum(a.nbytes for triple in cache.values() for a in triple) / 1e9
        logger.info(
            "Cached %d image triplets (%.2f GB) in %.0fs",
            len(cache),
            size_gb,
            time.perf_counter() - start,
        )
        return cache

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _load(path: Path, mode: str) -> np.ndarray:
        with Image.open(path) as img:
            return np.asarray(img.convert(mode))

    def _augment(
        self, image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the same random flip / 90-degree rotation to all three arrays."""
        if random.random() < 0.5:
            image_a, image_b, mask = image_a[:, ::-1], image_b[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image_a, image_b, mask = image_a[::-1], image_b[::-1], mask[::-1]
        k = random.randint(0, 3)
        if k:
            image_a, image_b, mask = (np.rot90(a, k) for a in (image_a, image_b, mask))
        return (
            np.ascontiguousarray(image_a),
            np.ascontiguousarray(image_b),
            np.ascontiguousarray(mask),
        )

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        array = image.astype(np.float32) / 255.0
        if self.normalize:
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        name, y, x = self.index[idx]
        size = self.tile_size
        # NOTE: the full image is decoded per tile. For 1024x1024 LEVIR-CD images
        # this is fast enough; pre-tile to disk if data loading becomes a bottleneck.
        if self._cache is not None:
            full_a, full_b, full_mask = self._cache[name]
        else:
            full_a = self._load(self.dir_a / name, "RGB")
            full_b = self._load(self.dir_b / name, "RGB")
            full_mask = self._load(self.dir_label / name, "L")
        image_a = extract_tile(full_a, y, x, size)
        image_b = extract_tile(full_b, y, x, size)
        mask = extract_tile(full_mask, y, x, size)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            image_a, image_b, mask = self._augment(image_a, image_b, mask)

        return {
            "image_a": self._to_tensor(image_a),
            "image_b": self._to_tensor(image_b),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "name": name,
            "y": y,
            "x": x,
        }
