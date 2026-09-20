"""Tile utilities for large satellite images (pure NumPy, no torch dependency)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def _axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Start positions along one axis so that the whole axis is covered."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] + tile_size < length:
        positions.append(length - tile_size)
    return positions


def compute_tile_grid(
    height: int, width: int, tile_size: int, overlap: int = 0
) -> list[tuple[int, int]]:
    """Return (y, x) top-left corners of tiles that fully cover a height x width image.

    The last row/column of tiles is shifted back so it ends exactly at the image
    border instead of producing a partially empty tile.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    stride = tile_size - overlap
    ys = _axis_positions(height, tile_size, stride)
    xs = _axis_positions(width, tile_size, stride)
    return [(y, x) for y in ys for x in xs]


def extract_tile(image: np.ndarray, y: int, x: int, tile_size: int) -> np.ndarray:
    """Crop a tile_size x tile_size window; zero-pad if the window exceeds the image."""
    tile = image[y : y + tile_size, x : x + tile_size]
    pad_h = tile_size - tile.shape[0]
    pad_w = tile_size - tile.shape[1]
    if pad_h or pad_w:
        pad = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2)
        tile = np.pad(tile, pad, mode="constant")
    return tile


def iter_tiles(
    image: np.ndarray, tile_size: int, overlap: int = 0
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield (y, x, tile) for every tile of an H x W (x C) array."""
    height, width = image.shape[:2]
    for y, x in compute_tile_grid(height, width, tile_size, overlap):
        yield y, x, extract_tile(image, y, x, tile_size)


def stitch_tiles(
    tiles: Sequence[np.ndarray],
    coords: Sequence[tuple[int, int]],
    out_hw: tuple[int, int],
) -> np.ndarray:
    """Merge tiles back into one array, averaging predictions where tiles overlap."""
    if not tiles:
        raise ValueError("tiles must not be empty")
    height, width = out_hw
    acc = np.zeros((height, width, *tiles[0].shape[2:]), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)
    for tile, (y, x) in zip(tiles, coords, strict=True):
        th = min(tile.shape[0], height - y)
        tw = min(tile.shape[1], width - x)
        acc[y : y + th, x : x + tw] += tile[:th, :tw]
        count[y : y + th, x : x + tw] += 1
    count = np.maximum(count, 1)
    if acc.ndim == 3:
        count = count[..., None]
    return acc / count
