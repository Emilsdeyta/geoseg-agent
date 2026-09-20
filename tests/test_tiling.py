import numpy as np
import pytest

from geoseg.data.tiling import compute_tile_grid, extract_tile, iter_tiles, stitch_tiles


def test_grid_covers_whole_image() -> None:
    height, width, size = 300, 517, 256
    coverage = np.zeros((height, width), dtype=int)
    for y, x in compute_tile_grid(height, width, size, overlap=32):
        coverage[y : y + size, x : x + size] += 1
    assert coverage.min() >= 1


def test_grid_exact_multiple_has_no_extra_tiles() -> None:
    assert len(compute_tile_grid(512, 512, 256)) == 4


def test_grid_image_smaller_than_tile() -> None:
    assert compute_tile_grid(100, 100, 256) == [(0, 0)]


@pytest.mark.parametrize("overlap", [-1, 256, 300])
def test_invalid_overlap(overlap: int) -> None:
    with pytest.raises(ValueError):
        compute_tile_grid(512, 512, 256, overlap)


def test_extract_tile_pads_small_images() -> None:
    image = np.ones((100, 120, 3), dtype=np.uint8)
    tile = extract_tile(image, 0, 0, 256)
    assert tile.shape == (256, 256, 3)
    assert tile[:100, :120].min() == 1
    assert tile[100:].max() == 0


def test_tile_then_stitch_roundtrip() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((300, 517, 3))
    tiles, coords = [], []
    for y, x, tile in iter_tiles(image, 256, overlap=64):
        tiles.append(tile)
        coords.append((y, x))
    restored = stitch_tiles(tiles, coords, (300, 517))
    np.testing.assert_allclose(restored, image)


def test_stitch_2d_masks() -> None:
    mask = np.zeros((300, 300))
    mask[100:200, 100:200] = 1.0
    tiles, coords = [], []
    for y, x, tile in iter_tiles(mask, 128, overlap=32):
        tiles.append(tile)
        coords.append((y, x))
    np.testing.assert_allclose(stitch_tiles(tiles, coords, (300, 300)), mask)
