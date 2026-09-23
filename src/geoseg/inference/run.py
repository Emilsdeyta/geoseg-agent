"""CLI: run full-image change-detection inference on a before/after image
pair and export the predicted change regions as a GeoJSON file.

Usage::

    python -m geoseg.inference.run \\
        --image-a before.png --image-b after.png \\
        --checkpoint outputs/full_run/best.pt \\
        --output result.geojson \\
        [--tile-size 256] [--overlap 0] [--threshold 0.5] [--min-area 0]

If --tile-size / --threshold are omitted, the values stored in the
checkpoint's own training config are used (same convention as
``geoseg.evaluation.evaluate`` / ``geoseg.evaluation.visualize``).

The output GeoJSON uses pixel coordinates, not a real-world CRS — see the
module docstring of ``geoseg.inference.polygonize`` for why (LEVIR-CD is
not georeferenced).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from geoseg.evaluation.evaluate import load_checkpoint_config
from geoseg.inference.polygonize import mask_to_polygons, polygons_to_geojson
from geoseg.inference.predict import predict_full_image
from geoseg.training.trainer import build_model

logger = logging.getLogger(__name__)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def run_inference(
    image_a_path: Path,
    image_b_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    tile_size: int | None = None,
    overlap: int = 0,
    threshold: float | None = None,
    min_area: float = 0.0,
) -> dict[str, Any]:
    """Load a checkpoint, run tiled inference on one image pair, and write
    the predicted change polygons to ``output_path`` as GeoJSON.

    Returns a small summary dict (also what the CLI prints to stdout).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = load_checkpoint_config(checkpoint, None)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    image_a = _load_rgb(image_a_path)
    image_b = _load_rgb(image_b_path)

    resolved_tile_size = tile_size if tile_size is not None else cfg.data.tile_size
    resolved_threshold = threshold if threshold is not None else cfg.train.threshold

    logger.info(
        "Running inference: %s vs %s (tile_size=%d, overlap=%d, threshold=%.3f)",
        image_a_path.name,
        image_b_path.name,
        resolved_tile_size,
        overlap,
        resolved_threshold,
    )
    mask = predict_full_image(
        model,
        image_a,
        image_b,
        tile_size=resolved_tile_size,
        overlap=overlap,
        device=device,
        threshold=resolved_threshold,
    )

    polygons = mask_to_polygons(mask, min_area=min_area)
    geojson = polygons_to_geojson(
        polygons,
        image_name=image_a_path.name,
        extra_properties={
            "checkpoint_epoch": checkpoint.get("epoch"),
            "threshold": resolved_threshold,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(geojson, indent=2))

    summary = {
        "image_a": str(image_a_path),
        "image_b": str(image_b_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "tile_size": resolved_tile_size,
        "overlap": overlap,
        "threshold": resolved_threshold,
        "min_area": min_area,
        "n_polygons": len(polygons),
        "changed_area_px": sum(p.area for p in polygons),
        "output": str(output_path),
    }
    logger.info("Wrote %d polygon(s) to %s", len(polygons), output_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run full-image change-detection inference and export a GeoJSON."
    )
    parser.add_argument("--image-a", type=Path, required=True, help="Path to the 'before' image.")
    parser.add_argument("--image-b", type=Path, required=True, help="Path to the 'after' image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .geojson path.")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Defaults to the value stored in the checkpoint's training config.",
    )
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Defaults to the value stored in the checkpoint's training config.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Drop predicted polygons smaller than this many pixel^2 (noise filtering).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    summary = run_inference(
        args.image_a,
        args.image_b,
        args.checkpoint,
        args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
        min_area=args.min_area,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
