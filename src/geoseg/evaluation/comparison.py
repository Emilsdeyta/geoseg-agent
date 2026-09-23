"""Pure (torch-free) helpers for building the best/worst comparison images.

Split out from ``visualize.py`` so these can be unit-tested without a torch install,
same as the rest of the test suite on CI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def rank_records(
    records: list[dict[str, Any]], n_best: int, n_worst: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sort by IoU ascending; return (best n_best, worst n_worst).

    If ``n_best + n_worst`` would overlap, the middle images are simply not selected by
    either slice (an image is never listed as both best and worst).
    """
    ordered = sorted(records, key=lambda r: r["iou"])
    worst = ordered[:n_worst]
    best = ordered[-n_best:] if n_best else []
    return best, worst


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.rectangle([xy, (xy[0] + 7 * len(text) + 6, xy[1] + 16)], fill=(0, 0, 0))
    draw.text((xy[0] + 3, xy[1] + 2), text, fill=(255, 255, 255))


def compose_comparison_image(
    image_a: np.ndarray,
    image_b: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    title: str,
    panel_size: int = 384,
) -> Image.Image:
    """image_a | image_b | ground truth | prediction | error map, with panel labels.

    Error map: green=TP, red=FP (false alarm), blue=FN (missed change), black=TN.
    """
    error = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    error[(pred_mask == 1) & (gt_mask == 1)] = (0, 200, 0)  # TP
    error[(pred_mask == 1) & (gt_mask == 0)] = (220, 0, 0)  # FP
    error[(pred_mask == 0) & (gt_mask == 1)] = (0, 80, 220)  # FN

    panels = [
        ("A (before)", image_a),
        ("B (after)", image_b),
        ("Ground truth", np.stack([gt_mask * 255] * 3, axis=-1)),
        ("Prediction", np.stack([pred_mask * 255] * 3, axis=-1)),
        ("Error (TP/FP/FN)", error),
    ]
    resized = [
        Image.fromarray(arr.astype(np.uint8)).resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        )
        for _, arr in panels
    ]

    gap = 6
    composite = Image.new(
        "RGB",
        (panel_size * len(panels) + gap * (len(panels) - 1), panel_size + 28),
        (30, 30, 30),
    )
    draw = ImageDraw.Draw(composite)
    draw.text((4, 4), title, fill=(255, 255, 255))
    for i, ((label, _), img) in enumerate(zip(panels, resized, strict=True)):
        x = i * (panel_size + gap)
        composite.paste(img, (x, 24))
        _label(draw, (x + 4, 28), label)
    return composite
