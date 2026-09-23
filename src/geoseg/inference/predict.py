"""Full-image inference: tile a pair of same-size images, run the model on
each tile pair, and stitch the tile-level probabilities back into a full
change mask.

This module requires torch and a trained SiameseUNet-style model (anything
with a ``forward(image_a, image_b) -> logits`` signature, where logits has
shape ``(B, 1, tile_size, tile_size)`` for one batch of tiles). It reuses
the same tiling/stitching approach already used and validated (by the
user, in their own venv) in ``geoseg.evaluation.visualize``:
``compute_tile_grid`` + ``extract_tile`` + ``stitch_tiles``. This means it
works on images of any size, not just exactly ``tile_size``, including a
full 1024x1024 LEVIR-CD pair or a larger real-world image tiled the same
way.

As with the rest of the torch-dependent code in this project, this module
cannot be executed end-to-end in the assistant's own sandbox (no local
torch install is feasible here — see project rules). Its Python-level
control flow (tiling loop, shape validation, thresholding, the
``return_prob`` branch) was exercised in the sandbox against a lightweight
stand-in for torch to check for logic errors, but real numerical
correctness against an actual trained model is only confirmed by running
``tests/test_predict.py`` in the user's own environment.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import torch

from geoseg.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from geoseg.data.tiling import compute_tile_grid, extract_tile, stitch_tiles

__all__ = ["predict_full_image"]


def _tile_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Same normalization as training/evaluation: scale to [0, 1], then
    ImageNet mean/std, then to a (C, H, W) float tensor."""
    array = image.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


@overload
def predict_full_image(
    model: torch.nn.Module,
    image_a: np.ndarray,
    image_b: np.ndarray,
    tile_size: int = 256,
    overlap: int = 0,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    return_prob: Literal[False] = False,
) -> np.ndarray: ...


@overload
def predict_full_image(
    model: torch.nn.Module,
    image_a: np.ndarray,
    image_b: np.ndarray,
    tile_size: int = 256,
    overlap: int = 0,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    *,
    return_prob: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...


def predict_full_image(
    model: torch.nn.Module,
    image_a: np.ndarray,
    image_b: np.ndarray,
    tile_size: int = 256,
    overlap: int = 0,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    return_prob: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run change-detection inference on a full-size before/after image pair.

    Parameters
    ----------
    model:
        A trained model with ``forward(image_a, image_b) -> logits`` where
        logits has shape ``(1, 1, tile_size, tile_size)`` for a single
        tile. This function calls ``model.eval()`` itself; the caller only
        needs to have already moved the model to ``device``.
    image_a, image_b:
        ``(H, W, 3)`` uint8 RGB arrays with matching shapes. Not limited to
        any particular size — larger images are tiled automatically.
    tile_size:
        Tile edge length the model was trained on (LEVIR-CD default: 256).
    overlap:
        Overlap between adjacent tiles in pixels, forwarded to
        ``compute_tile_grid``. ``0`` (the default) matches how the
        existing evaluation/visualization code tiles full images; a
        positive value can smooth boundary seams at the cost of more
        tiles to run (``stitch_tiles`` averages overlapping predictions).
    device:
        Torch device (or device string) to run inference on.
    threshold:
        Probability threshold applied to the stitched sigmoid output to
        obtain the binary mask.
    return_prob:
        If True, also return the stitched ``(H, W)`` float32 probability
        map alongside the binary mask.

    Returns
    -------
    A ``(H, W)`` uint8 array of 0/1 values (1 = predicted change). If
    ``return_prob`` is True, returns ``(mask, prob)`` instead, where
    ``prob`` is the ``(H, W)`` float32 probability map before thresholding.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"image_a and image_b must have the same shape, got {image_a.shape} and {image_b.shape}"
        )
    if image_a.ndim != 3 or image_a.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB images, got shape {image_a.shape}")

    resolved_device = torch.device(device) if isinstance(device, str) else device
    height, width = image_a.shape[:2]
    coords = compute_tile_grid(height, width, tile_size, overlap=overlap)

    model.eval()
    prob_tiles = []
    with torch.no_grad():
        for y, x in coords:
            tile_a = _tile_to_tensor(extract_tile(image_a, y, x, tile_size)).unsqueeze(0)
            tile_b = _tile_to_tensor(extract_tile(image_b, y, x, tile_size)).unsqueeze(0)
            logits = model(tile_a.to(resolved_device), tile_b.to(resolved_device))
            prob_tiles.append(torch.sigmoid(logits)[0, 0].cpu().numpy())

    prob_full = stitch_tiles(prob_tiles, coords, (height, width))
    mask = (prob_full > threshold).astype(np.uint8)

    if return_prob:
        return mask, prob_full.astype(np.float32)
    return mask
