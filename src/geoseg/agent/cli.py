"""CLI: generate a natural-language report from an already-produced
GeoJSON (and, optionally, a test_metrics.json), with no LLM/API key needed.

Usage::

    python -m geoseg.agent.cli --geojson result.geojson \\
        [--metrics outputs/full_run/test_metrics.json] \\
        [--image-name my_image.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoseg.agent.report import generate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a rule-based natural-language report from a GeoJSON."
    )
    parser.add_argument("--geojson", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None, help="Optional test_metrics.json")
    parser.add_argument("--image-name", default=None, help="Override the image name shown.")
    parser.add_argument("--no-caveat", action="store_true", help="Omit the reliability caveat.")
    args = parser.parse_args(argv)

    geojson = json.loads(args.geojson.read_text())
    metrics = json.loads(args.metrics.read_text()) if args.metrics else None

    report = generate_report(
        geojson,
        metrics=metrics,
        image_name=args.image_name,
        include_caveat=not args.no_caveat,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
