from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from geoseg.inference.polygonize import mask_to_polygons, polygons_to_geojson


def _empty_mask(h: int = 20, w: int = 20) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def test_empty_mask_returns_no_polygons() -> None:
    mask = _empty_mask()
    assert mask_to_polygons(mask) == []


def test_full_mask_returns_one_polygon_covering_whole_area() -> None:
    mask = np.ones((20, 20), dtype=np.uint8)
    polys = mask_to_polygons(mask)
    assert len(polys) == 1
    # Marching-squares tracing chamfers the 4 corners slightly, so area is a
    # little under the exact 400 px; allow a small, known slack for that.
    assert polys[0].area == pytest.approx(400, rel=0.03)


def test_single_square_blob_has_expected_area() -> None:
    mask = _empty_mask(30, 30)
    mask[5:15, 5:15] = 1  # 10x10 square = 100 px
    polys = mask_to_polygons(mask)
    assert len(polys) == 1
    assert polys[0].area == pytest.approx(100, rel=0.03)
    assert polys[0].is_valid


def test_two_disjoint_blobs_return_two_polygons() -> None:
    mask = _empty_mask(30, 30)
    mask[2:6, 2:6] = 1
    mask[20:26, 20:26] = 1
    polys = mask_to_polygons(mask)
    assert len(polys) == 2


def test_donut_shape_produces_polygon_with_hole() -> None:
    mask = _empty_mask(30, 30)
    mask[5:25, 5:25] = 1
    mask[12:18, 12:18] = 0  # carve out a hole in the middle
    polys = mask_to_polygons(mask)
    assert len(polys) == 1
    poly = polys[0]
    assert len(poly.interiors) == 1
    # Area should be roughly (outer 20x20) - (inner 6x6) = 400 - 36 = 364
    assert poly.area == pytest.approx(364, rel=0.05)


def test_min_area_filters_small_blobs() -> None:
    mask = _empty_mask(30, 30)
    mask[2:4, 2:4] = 1  # tiny 2x2 = 4 px blob (noise)
    mask[10:20, 10:20] = 1  # real 10x10 = 100 px blob

    all_polys = mask_to_polygons(mask, min_area=0.0)
    assert len(all_polys) == 2

    filtered = mask_to_polygons(mask, min_area=50.0)
    assert len(filtered) == 1
    assert filtered[0].area == pytest.approx(100, rel=0.03)


def test_mask_must_be_2d() -> None:
    with pytest.raises(ValueError):
        mask_to_polygons(np.zeros((2, 3, 3), dtype=np.uint8))


def test_polygons_to_geojson_structure() -> None:
    mask = _empty_mask(20, 20)
    mask[2:8, 2:8] = 1
    polys = mask_to_polygons(mask)

    geojson = polygons_to_geojson(polys, image_name="test_01.png")

    assert geojson["type"] == "FeatureCollection"
    assert geojson["crs"]["properties"]["name"] == "local-pixel"
    assert len(geojson["features"]) == 1

    feature = geojson["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["image_name"] == "test_01.png"
    assert feature["properties"]["area_px"] == pytest.approx(36, rel=0.06)


def test_polygons_to_geojson_empty_list() -> None:
    geojson = polygons_to_geojson([])
    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"] == []


def test_polygons_to_geojson_extra_properties_merged() -> None:
    mask = _empty_mask(20, 20)
    mask[2:8, 2:8] = 1
    polys = mask_to_polygons(mask)

    geojson = polygons_to_geojson(
        polys, image_name="t.png", extra_properties={"checkpoint_epoch": 42}
    )
    props = geojson["features"][0]["properties"]
    assert props["checkpoint_epoch"] == 42
    assert props["image_name"] == "t.png"


@pytest.mark.slow
def test_many_small_components_stays_fast() -> None:
    """Regression guard: a raw, un-cleaned prediction mask can have thousands
    of isolated single-pixel false positives. The naive O(n^2) nesting-depth
    check (before the bounding-box prefilter + cached .area) took ~40s for
    3,000 such components on this machine; this test fails loudly (via
    timeout) if that regresses instead of silently becoming slow in CI.
    """
    rng = np.random.default_rng(1)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    ys = rng.integers(0, 1024, size=3000)
    xs = rng.integers(0, 1024, size=3000)
    mask[ys, xs] = 1

    import time

    start = time.time()
    polys = mask_to_polygons(mask)
    elapsed = time.time() - start

    assert len(polys) > 2500  # most random single/adjacent pixels stay separate
    assert elapsed < 5.0, f"mask_to_polygons took {elapsed:.1f}s, expected < 5s"


def test_returned_polygons_are_valid_shapely_polygons() -> None:
    mask = _empty_mask(20, 20)
    mask[2:8, 2:8] = 1
    polys = mask_to_polygons(mask)
    assert all(isinstance(p, Polygon) for p in polys)
    assert all(p.is_valid for p in polys)
