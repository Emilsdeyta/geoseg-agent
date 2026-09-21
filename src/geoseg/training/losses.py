"""Losses for binary change detection.

Only ~5% of the pixels in LEVIR-CD are "changed", so plain BCE is dominated by
the background. Dice looks at overlap of the changed class directly and Focal
down-weights easy background pixels; their sum is the default training loss.

All losses take raw logits and a float target in {0, 1}, both shaped (B, 1, H, W).
"""

from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Soft Dice loss computed over the whole batch (not per sample)."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits).flatten()
        target = target.flatten()
        intersection = (prob * target).sum()
        denominator = prob.sum() + target.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)


class FocalLoss(nn.Module):
    """Binary focal loss: ``(1 - p_t) ** gamma * BCE``.

    ``alpha`` (weight of the positive class) is off by default: Dice already
    handles the imbalance and alpha < 0.5 would further down-weight the rare class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        loss = (1.0 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            loss = alpha_t * loss
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """``dice_weight * Dice + focal_weight * Focal``."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        gamma: float = 2.0,
        alpha: float | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth)
        self.focal = FocalLoss(gamma, alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.focal_weight * self.focal(
            logits, target
        )


LOSS_NAMES = ("bce", "dice", "focal", "dice_focal")


def build_loss(name: str, gamma: float = 2.0, alpha: float | None = None) -> nn.Module:
    """Create a loss by name (used for the BCE / Dice / Focal ablation)."""
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "focal":
        return FocalLoss(gamma, alpha)
    if name == "dice_focal":
        return DiceFocalLoss(gamma=gamma, alpha=alpha)
    raise ValueError(f"Unknown loss {name!r}, expected one of {LOSS_NAMES}")
