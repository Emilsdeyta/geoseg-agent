"""Command line entry point.

    python -m geoseg.training.train --config configs/default.yaml
    python -m geoseg.training.train --config configs/default.yaml \\
        --set train.epochs=2 data.root=/kaggle/input/levir-cd

    # continue an interrupted run (auto = resume from <output_dir>/last.pt if present)
    python -m geoseg.training.train --config configs/default.yaml --resume auto
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from geoseg.training.config import load_config
from geoseg.training.trainer import fit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the change detection model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="override config values, e.g. train.epochs=5 data.num_workers=0",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="'auto' to resume from <output_dir>/last.pt if it exists, "
        "or an explicit checkpoint path. Default: start fresh.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Third-party HTTP clients log every request at INFO (e.g. pretrained weight downloads).
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config, args.overrides)
    summary = fit(cfg, resume=args.resume)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
