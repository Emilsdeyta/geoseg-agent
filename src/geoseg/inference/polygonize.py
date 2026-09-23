"""Convert binary change masks into vector polygons and GeoJSON.

This module intentionally has no dependency on torch (or any part of the
training/model code), so it can be fully unit-tested without a GPU or a
PyTorch install. It operates purely on NumPy arrays.

Coordinate system note
-----------------------
LEVIR-CD images are not georeferenced (no CRS, no affine transform). The
polygons and GeoJSON produced here therefore use **pixel coordinates**:
x = column index, y = row index, both in the source mask's pixel grid, with
(0, 0) at the top-left corner. This is *not* a real-world CRS (not lat/lon).
The GeoJSON emitted here marks this explicitly via a non-standard "crs"
member (name "local-pixel") so downstream consumers are not misled into
treating the coordinates as WGS84 degrees. If a real georeferenced input
(e.g. a GeoTIFF with an affine transform) becomes available later, an
`affine_transform` argument can be added to convert pixel coordinates to
real-world coordinates before calling `polygons_to_geojson`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon, mapping
from skimage import measure

__all__ = ["mask_to_polygons", "polygons_to_geojson"]


def _raw_contour_polygons(binary: np.ndarray) -> list[Polygon]:
    """Trace all 0/1 boundaries in a binary mask as raw polygons.

    Uses marching squares at the 0.5 level (`skimage.measure.find_contours`)
    rather than OpenCV's `findContours`, because marching squares places the
    boundary *between* foreground and background pixels. This keeps the
    resulting polygon's area close to the true pixel count (e.g. a solid
    10x10 block of pixels yields an area close to 100), whereas tracing
    pixel centers directly (OpenCV's approach) would shrink a solid NxN
    block down to (N-1)x(N-1).

    The input is padded by one pixel of background on every side before
    tracing, so that regions touching the array border still produce a
    closed boundary (marching squares needs a 0/1 transition, which does
    not exist along an edge that is entirely foreground). Returned polygon
    coordinates are already shifted back to the original (unpadded) pixel
    grid.
    """
    padded = np.pad(binary, pad_width=1, mode="constant", constant_values=0)
    raw_contours = measure.find_contours(padded.astype(float), level=0.5)

    polygons: list[Polygon] = []
    for contour in raw_contours:
        if len(contour) < 4:  # a closed ring needs at least 3 unique points
            continue
        # skimage returns (row, col); shift by -1 to undo the padding, and
        # swap to (x, y) = (col, row) for conventional GIS-style coordinates.
        points = [(float(c) - 1.0, float(r) - 1.0) for r, c in contour]
        try:
            poly = Polygon(points)
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area == 0:
            continue
        if isinstance(poly, Polygon):
            polygons.append(poly)
    return polygons


def mask_to_polygons(mask: np.ndarray, min_area: float = 0.0) -> list[Polygon]:
    """Convert a binary 2D mask into a list of Shapely polygons.

    Parameters
    ----------
    mask:
        2D array (H, W). Any nonzero value is treated as foreground
        ("changed"); zero is background. Must be 2-dimensional.
    min_area:
        Minimum polygon area, in pixel^2 units, for a polygon to be kept.
        Use this to drop noise-sized blobs (e.g. single stray pixels).
        Defaults to 0.0 (keep everything with nonzero area).

    Returns
    -------
    A list of valid, non-empty `shapely.geometry.Polygon` objects. Holes in
    the mask (background regions fully enclosed by foreground) are
    preserved as interior rings, one nesting level at a time (a hole inside
    a hole becomes its own separate exterior polygon, matching how a real
    "island in a lake in a landmass" would be represented). Polygons that
    become empty or invalid after cleanup are dropped.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    binary = (mask != 0).astype(np.uint8)
    if not binary.any():
        return []

    raw_polys = _raw_contour_polygons(binary)
    if not raw_polys:
        return []

    n = len(raw_polys)
    points = [poly.representative_point() for poly in raw_polys]
    bounds = [poly.bounds for poly in raw_polys]  # (minx, miny, maxx, maxy)
    areas = [poly.area for poly in raw_polys]  # cache: repeated .area calls are the
    # dominant cost in the loop below (each is a GEOS call, not a cheap attribute read)

    # Nesting depth: how many other (larger) raw rings contain this ring's
    # representative point. Even depth = an exterior boundary; odd depth =
    # a hole cut out of the nearest enclosing exterior.
    #
    # A cheap bounding-box test is applied before the exact (and much more
    # expensive) `Polygon.contains` check. This matters in practice: a raw,
    # un-cleaned model prediction can contain thousands of isolated
    # single-pixel false positives, and without this prefilter the naive
    # O(n^2) exact-geometry check becomes the dominant cost (tens of
    # seconds for a few thousand components, measured directly). With the
    # bounding-box prefilter, disjoint specks are rejected in O(1) and the
    # expensive check only runs for rings whose bounding boxes actually
    # overlap, which is the rare case for typical change masks.
    depths = [0] * n
    for i in range(n):
        px, py = points[i].x, points[i].y
        area_i = areas[i]
        for j in range(n):
            if i == j or areas[j] <= area_i:
                continue
            minx, miny, maxx, maxy = bounds[j]
            if not (minx <= px <= maxx and miny <= py <= maxy):
                continue
            if raw_polys[j].contains(points[i]):
                depths[i] += 1

    exterior_idx = [i for i in range(n) if depths[i] % 2 == 0]
    hole_idx = [i for i in range(n) if depths[i] % 2 == 1]

    holes_by_parent: dict[int, list[list[tuple[float, float]]]] = {i: [] for i in exterior_idx}
    for h in hole_idx:
        pt: Point = points[h]
        containing = [(i, raw_polys[i]) for i in exterior_idx if raw_polys[i].contains(pt)]
        if not containing:
            continue  # orphaned hole ring (shouldn't normally happen); drop it
        parent_i, _ = min(containing, key=lambda pair: pair[1].area)
        holes_by_parent[parent_i].append(list(raw_polys[h].exterior.coords))

    polygons: list[Polygon] = []
    for i in exterior_idx:
        holes = holes_by_parent[i]
        try:
            poly = Polygon(raw_polys[i].exterior.coords, holes=holes if holes else None)
        except Exception:
            continue

        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        if isinstance(poly, Polygon):
            candidates = [poly]
        else:
            # buffer(0) on a self-intersecting ring can yield a MultiPolygon.
            candidates = [g for g in getattr(poly, "geoms", []) if isinstance(g, Polygon)]

        for geom in candidates:
            if not geom.is_empty and geom.area >= min_area:
                polygons.append(geom)

    return polygons


def polygons_to_geojson(
    polygons: list[Polygon],
    image_name: str | None = None,
    extra_properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap a list of polygons into a GeoJSON FeatureCollection.

    The coordinates are pixel coordinates, not a real-world CRS — see the
    module docstring. Each feature gets a `properties.area_px` field (the
    polygon's area in pixel^2) plus, if given, `image_name` and any
    `extra_properties` merged in.

    Parameters
    ----------
    polygons:
        Polygons as returned by `mask_to_polygons`.
    image_name:
        Optional source image identifier, stored on every feature.
    extra_properties:
        Optional extra key/value pairs merged into every feature's
        properties (e.g. {"model_checkpoint_epoch": 42}).

    Returns
    -------
    A GeoJSON-compatible dict (FeatureCollection).
    """
    features = []
    for poly in polygons:
        properties: dict[str, Any] = {"area_px": poly.area}
        if image_name is not None:
            properties["image_name"] = image_name
        if extra_properties:
            properties.update(extra_properties)

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": properties,
            }
        )

    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "local-pixel",
                "note": (
                    "Coordinates are pixel units (x=column, y=row) in the "
                    "source image's own grid, NOT a real-world CRS such as "
                    "WGS84. This dataset is not georeferenced."
                ),
            },
        },
        "features": features,
    }
