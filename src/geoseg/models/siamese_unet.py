"""Siamese U-Net for bi-temporal change detection.

Both acquisition dates go through the *same* encoder (shared weights). At every
resolution level the two feature maps are fused (absolute difference or
concatenation) and the fused maps act as the skip connections of a U-Net decoder.
The output is one logit per pixel: change / no change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn

Fusion = Literal["diff", "concat"]


class ConvBNReLU(nn.Sequential):
    """3x3 convolution + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample to a target size, concatenate the skip connection, refine with two convs."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(
        self, x: torch.Tensor, size: list[int], skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=size, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SiameseUNet(nn.Module):
    """Change detection network: ``forward(image_a, image_b) -> logits (B, 1, H, W)``.

    Args:
        encoder_name: any encoder name supported by ``segmentation_models_pytorch``.
        encoder_weights: ``"imagenet"`` to start from pretrained weights (needs internet
            on first use) or ``None`` for random initialisation.
        decoder_channels: channels of each decoder block; its length is the encoder depth.
        fusion: ``"diff"`` uses ``|f_a - f_b|``; ``"concat"`` uses ``[f_a, f_b]``.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        fusion: Fusion = "diff",
    ) -> None:
        super().__init__()
        if fusion not in ("diff", "concat"):
            raise ValueError(f"fusion must be 'diff' or 'concat', got {fusion!r}")
        self.fusion = fusion

        depth = len(decoder_channels)
        self.encoder = get_encoder(
            encoder_name, in_channels=3, depth=depth, weights=encoder_weights
        )

        # encoder.out_channels = [3, c1, ..., c_depth]; level 0 is the raw input and is skipped.
        multiplier = 2 if fusion == "concat" else 1
        fused_channels = [c * multiplier for c in self.encoder.out_channels[1:]]

        head_channels = fused_channels[-1]  # deepest level, decoder input
        skip_channels = [*reversed(fused_channels[:-1]), 0]  # last block has no skip
        in_channels = [head_channels, *decoder_channels[:-1]]
        self.blocks = nn.ModuleList(
            DecoderBlock(i, s, o)
            for i, s, o in zip(in_channels, skip_channels, decoder_channels, strict=True)
        )
        self.head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def _fuse(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        if self.fusion == "diff":
            return torch.abs(feat_a - feat_b)
        return torch.cat([feat_a, feat_b], dim=1)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        # One pass over the stacked batch = shared weights, and BatchNorm sees both dates.
        features = self.encoder(torch.cat([image_a, image_b], dim=0))
        pairs = [f.chunk(2, dim=0) for f in features[1:]]
        fused = [self._fuse(fa, fb) for fa, fb in pairs]

        x = fused[-1]
        skips = fused[:-1][::-1]
        sizes = [[int(s) for s in f.shape[-2:]] for f in skips]
        sizes.append([int(s) for s in image_a.shape[-2:]])  # last block restores input size
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, sizes[i], skip)
        return self.head(x)
