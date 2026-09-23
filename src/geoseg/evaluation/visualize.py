"""Export side-by-side (image_a | image_b | ground truth | prediction | error map)
comparisons for the best- and worst-scoring images in a split, ranked by per-image IoU.

Unlike ``evaluate.py`` (tile-level metrics, matching training), this reconstructs each
*full* image from its tiles (stitching predicted probabilities back together) so the
comparison images show a whole 1024x1024 LEVIR-CD pair, not an isolated 256x256 tile.

Usage::

    python -m geoseg.evaluation.visualize --checkpoint outputs/full_run/best.pt \\
        --data-root /kaggle/working/levir-cd --split test \\
        --output-dir outputs/full_run/error_analysis --n-best 5 --n-worst 5

Writes ``ranking.json`` (every image's IoU/F1, sorted worst to best) plus one PNG per
selected image. The error map colours: green = correctly flagged change (TP), red =
false alarm (FP), blue = missed change (FN), black = correctly flagged no-change (TN).
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

from geoseg.data.dataset import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD
from geoseg.data.tiling import compute_tile_grid, extract_tile, stitch_tiles
from geoseg.evaluation.comparison import compose_comparison_image, rank_records
from geoseg.evaluation.evaluate import load_checkpoint_config
from geoseg.training.trainer import build_model

logger = logging.getLogger(__name__)


def _load_image(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert(mode))


def _tile_to_tensor(image: np.ndarray) -> torch.Tensor:
    array = image.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def compute_full_image_records(
    model: torch.nn.Module,
    root: str | Path,
    split: str,
    tile_size: int,
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    """One record per image: name, iou, f1, and the full-size arrays needed to plot it."""
    split_dir = Path(root) / split
    dir_a, dir_b, dir_label = split_dir / "A", split_dir / "B", split_dir / "label"
    names = sorted(p.name for p in dir_a.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    model.eval()
    records = []
    with torch.no_grad():
        for name in names:
            full_a = _load_image(dir_a / name, "RGB")
            full_b = _load_image(dir_b / name, "RGB")
            full_label = _load_image(dir_label / name, "L")
            height, width = full_label.shape
            coords = compute_tile_grid(height, width, tile_size, overlap=0)

            prob_tiles = []
            for y, x in coords:
                tensor_a = _tile_to_tensor(extract_tile(full_a, y, x, tile_size))
                tensor_b = _tile_to_tensor(extract_tile(full_b, y, x, tile_size))
                logits = model(tensor_a.unsqueeze(0).to(device), tensor_b.unsqueeze(0).to(device))
                prob_tiles.append(torch.sigmoid(logits)[0, 0].cpu().numpy())
            prob_full = stitch_tiles(prob_tiles, coords, (height, width))

            gt_mask = (full_label > 127).astype(np.uint8)
            pred_mask = (prob_full > threshold).astype(np.uint8)
            tp = int(((pred_mask == 1) & (gt_mask == 1)).sum())
            fp = int(((pred_mask == 1) & (gt_mask == 0)).sum())
            fn = int(((pred_mask == 0) & (gt_mask == 1)).sum())
            denom_iou, denom_f1 = tp + fp + fn, 2 * tp + fp + fn
            records.append(
                {
                    "name": name,
                    "iou": tp / denom_iou if denom_iou else 1.0,
                    "f1": 2 * tp / denom_f1 if denom_f1 else 1.0,
                    "image_a": full_a,
                    "image_b": full_b,
                    "gt_mask": gt_mask,
                    "pred_mask": pred_mask,
                }
            )
    return records


def export_error_analysis(
    checkpoint_path: Path,
    *,
    data_root: str | None,
    split: str,
    output_dir: Path,
    n_best: int,
    n_worst: int,
    threshold: float | None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = load_checkpoint_config(checkpoint, data_root)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    records = compute_full_image_records(
        model,
        cfg.data.root,
        split,
        cfg.data.tile_size,
        device,
        threshold if threshold is not None else cfg.train.threshold,
    )
    best, worst = rank_records(records, n_best, n_worst)

    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, group in (("worst", worst), ("best", best)):
        for rank, record in enumerate(group, start=1):
            title = (
                f"{kind} #{rank}  {record['name']}  iou={record['iou']:.3f} f1={record['f1']:.3f}"
            )
            image = compose_comparison_image(
                record["image_a"],
                record["image_b"],
                record["gt_mask"],
                record["pred_mask"],
                title=title,
            )
            image.save(output_dir / f"{kind}_{rank:02d}_{record['name']}")

    ranking = sorted(
        ({"name": r["name"], "iou": r["iou"], "f1": r["f1"]} for r in records),
        key=lambda r: r["iou"],
    )
    (output_dir / "ranking.json").write_text(json.dumps(ranking, indent=2))
    return {
        "split": split,
        "n_images": len(records),
        "mean_iou": sum(r["iou"] for r in records) / len(records),
        "exported_best": len(best),
        "exported_worst": len(worst),
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export best/worst prediction comparisons.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis"))
    parser.add_argument("--n-best", type=int, default=5)
    parser.add_argument("--n-worst", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    summary = export_error_analysis(
        args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        output_dir=args.output_dir,
        n_best=args.n_best,
        n_worst=args.n_worst,
        threshold=args.threshold,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
