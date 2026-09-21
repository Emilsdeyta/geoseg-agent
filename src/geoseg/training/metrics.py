"""Segmentation metrics for the *change* class.

Accuracy is useless here (predicting "no change" everywhere already scores ~95%),
so we report IoU, F1, precision and recall of the change class. Counts are
accumulated over the whole dataset and the metrics computed once at the end;
averaging per-batch metrics would be biased by tiles without any change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class BinaryConfusion:
    """Running confusion-matrix counts for a binary mask."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        """Add a batch. ``logits`` are raw model outputs, ``target`` is {0, 1}."""
        pred = torch.sigmoid(logits) > threshold
        truth = target > 0.5
        self.tp += int((pred & truth).sum())
        self.fp += int((pred & ~truth).sum())
        self.fn += int((~pred & truth).sum())
        self.tn += int((~pred & ~truth).sum())

    def compute(self) -> dict[str, float]:
        """IoU, F1, precision, recall of the positive class (0.0 when undefined)."""
        return {
            "iou": _safe_div(self.tp, self.tp + self.fp + self.fn),
            "f1": _safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "precision": _safe_div(self.tp, self.tp + self.fp),
            "recall": _safe_div(self.tp, self.tp + self.fn),
        }
