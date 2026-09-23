"""Tests for geoseg.inference.predict.

These tests require torch and therefore cannot be run in the assistant's
own sandbox (see project rules on torch-dependent code). They were only
confirmed by exercising predict.py's control flow against a lightweight,
non-shipped stand-in for torch; real numerical behavior against this exact
test file is confirmed by running it here, in the real environment.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geoseg.inference.predict import predict_full_image  # noqa: E402


class _ConstantLogitModel(torch.nn.Module):
    """A trivial model that ignores its inputs and always outputs the same
    logit value everywhere, so the resulting sigmoid probability (and thus
    the thresholded mask) is fully predictable."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image_a.shape
        return torch.full((b, 1, h, w), self.value)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _random_rgb(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_high_logit_gives_all_change_mask(rng: np.random.Generator) -> None:
    image_a, image_b = _random_rgb(rng, 64, 64), _random_rgb(rng, 64, 64)
    model = _ConstantLogitModel(10.0)  # sigmoid(10) ~= 0.9999
    mask = predict_full_image(model, image_a, image_b, tile_size=32, threshold=0.5)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    assert (mask == 1).all()


def test_low_logit_gives_all_no_change_mask(rng: np.random.Generator) -> None:
    image_a, image_b = _random_rgb(rng, 64, 64), _random_rgb(rng, 64, 64)
    model = _ConstantLogitModel(-10.0)  # sigmoid(-10) ~= 0.0000454
    mask = predict_full_image(model, image_a, image_b, tile_size=32, threshold=0.5)
    assert (mask == 0).all()


def test_threshold_boundary_behavior(rng: np.random.Generator) -> None:
    image_a, image_b = _random_rgb(rng, 64, 64), _random_rgb(rng, 64, 64)
    model = _ConstantLogitModel(0.0)  # sigmoid(0) == 0.5 exactly

    below = predict_full_image(model, image_a, image_b, tile_size=32, threshold=0.4)
    above = predict_full_image(model, image_a, image_b, tile_size=32, threshold=0.6)

    assert (below == 1).all()
    assert (above == 0).all()


def test_return_prob_matches_expected_sigmoid(rng: np.random.Generator) -> None:
    image_a, image_b = _random_rgb(rng, 64, 64), _random_rgb(rng, 64, 64)
    model = _ConstantLogitModel(0.0)

    mask, prob = predict_full_image(
        model, image_a, image_b, tile_size=32, threshold=0.5, return_prob=True
    )
    assert prob.shape == (64, 64)
    assert prob.dtype == np.float32
    assert np.allclose(prob, 0.5, atol=1e-4)
    assert isinstance(mask, np.ndarray)


def test_mismatched_shapes_raise_value_error(rng: np.random.Generator) -> None:
    image_a = _random_rgb(rng, 64, 64)
    image_b = _random_rgb(rng, 64, 32)
    model = _ConstantLogitModel(1.0)
    with pytest.raises(ValueError, match="same shape"):
        predict_full_image(model, image_a, image_b, tile_size=32)


def test_non_rgb_input_raises_value_error(rng: np.random.Generator) -> None:
    image_a = _random_rgb(rng, 64, 64)[:, :, 0]  # drop to (H, W), no channel dim
    image_b = _random_rgb(rng, 64, 64)[:, :, 0]
    model = _ConstantLogitModel(1.0)
    with pytest.raises(ValueError, match="RGB"):
        predict_full_image(model, image_a, image_b, tile_size=32)


def test_image_size_not_a_multiple_of_tile_size(rng: np.random.Generator) -> None:
    """Exercises the edge-tile path in compute_tile_grid/extract_tile/stitch_tiles
    end-to-end through predict_full_image, using a size that doesn't divide
    evenly by tile_size (mirrors a real, non-1024-aligned input image)."""
    image_a, image_b = _random_rgb(rng, 70, 70), _random_rgb(rng, 70, 70)
    model = _ConstantLogitModel(10.0)
    mask = predict_full_image(model, image_a, image_b, tile_size=32, threshold=0.5)
    assert mask.shape == (70, 70)
    assert (mask == 1).all()


def test_overlap_does_not_change_result_for_constant_model(rng: np.random.Generator) -> None:
    """With a model that outputs the same value everywhere, averaging
    overlapping tile predictions should not change the result — this
    isolates whether the overlap path runs correctly, independent of any
    real model's tile-boundary behavior."""
    image_a, image_b = _random_rgb(rng, 64, 64), _random_rgb(rng, 64, 64)
    model = _ConstantLogitModel(10.0)
    mask_no_overlap = predict_full_image(model, image_a, image_b, tile_size=32, overlap=0)
    mask_overlap = predict_full_image(model, image_a, image_b, tile_size=32, overlap=8)
    assert np.array_equal(mask_no_overlap, mask_overlap)


def test_accepts_device_as_string_or_torch_device(rng: np.random.Generator) -> None:
    image_a, image_b = _random_rgb(rng, 32, 32), _random_rgb(rng, 32, 32)
    model = _ConstantLogitModel(10.0)
    mask_str = predict_full_image(model, image_a, image_b, tile_size=32, device="cpu")
    mask_dev = predict_full_image(model, image_a, image_b, tile_size=32, device=torch.device("cpu"))
    assert np.array_equal(mask_str, mask_dev)
