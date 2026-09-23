"""Draw predicted change polygons (from a GeoJSON produced by
geoseg.inference.run) on top of an image, as a quick visual sanity check.

This is intentionally standalone and has no torch dependency — it only
needs the image and the already-produced GeoJSON, so it can be run right
after `python -m geoseg.inference.run` to eyeball whether the polygons
line up with real change on the ground (not, say, transposed x/y, or
offset by a tile boundary).

Usage::

    python scripts/overlay_geojson.py --image data/raw/levir-cd/test/B/test_1.png \\
        --geojson test_result.geojson --output overlay.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def draw_overlay(image_path: Path, geojson_path: Path, output_path: Path) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    geojson = json.loads(geojson_path.read_text())
    n_drawn = 0
    for feature in geojson["features"]:
        geom = feature["geometry"]
        if geom["type"] != "Polygon":
            continue
        exterior = geom["coordinates"][0]
        draw.polygon(
            [tuple(point) for point in exterior],
            outline=(255, 0, 0, 255),
            fill=(255, 0, 0, 60),
            width=2,
        )
        n_drawn += 1

    image.save(output_path)
    return n_drawn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Overlay a prediction GeoJSON on its image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    n_drawn = draw_overlay(args.image, args.geojson, args.output)
    print(f"Drew {n_drawn} polygon(s) onto {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
