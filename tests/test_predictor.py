"""TorchPredictor wiring, tested with fake torch/model modules (no real torch needed)."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from geoseg.api.predictor import TorchPredictor


class _FakeModel:
    def __init__(self) -> None:
        self.loaded_state: Any = None
        self.eval_called = False

    def to(self, device: Any) -> _FakeModel:
        return self

    def load_state_dict(self, state: Any) -> None:
        self.loaded_state = state

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"model": _FakeModel()}

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    fake_torch.device = lambda name: name  # type: ignore[attr-defined]
    fake_torch.load = lambda path, map_location=None: {  # type: ignore[attr-defined]
        "model": {"w": 1},
        "epoch": 7,
    }
    fake_torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]

    def load_cfg(checkpoint: Any, override: Any) -> Any:
        return SimpleNamespace(
            model=SimpleNamespace(encoder_weights="imagenet"),
            data=SimpleNamespace(tile_size=256),
            train=SimpleNamespace(threshold=0.42),
        )

    def build_model(cfg: Any) -> _FakeModel:
        seen["weights_at_build"] = cfg.model.encoder_weights
        return seen["model"]  # type: ignore[no-any-return]

    def predict_full_image(model: Any, a: Any, b: Any, **kwargs: Any) -> np.ndarray:
        seen["predict_kwargs"] = kwargs
        return np.ones(a.shape[:2], dtype=bool)

    for name, attrs in {
        "torch": fake_torch,
        "geoseg.evaluation.evaluate": SimpleNamespace(load_checkpoint_config=load_cfg),
        "geoseg.training.trainer": SimpleNamespace(build_model=build_model),
        "geoseg.inference.predict": SimpleNamespace(predict_full_image=predict_full_image),
    }.items():
        monkeypatch.setitem(sys.modules, name, attrs)
    return seen


def test_encoder_weights_disabled_before_build(fakes: dict[str, Any]) -> None:
    TorchPredictor(Path("best.pt"))
    assert fakes["weights_at_build"] is None


def test_loads_state_and_metadata(fakes: dict[str, Any]) -> None:
    pred = TorchPredictor(Path("best.pt"))
    assert fakes["model"].loaded_state == {"w": 1}
    assert fakes["model"].eval_called is True
    assert pred.checkpoint_epoch == 7
    assert pred.default_threshold == pytest.approx(0.42)


def test_predict_mask_uses_default_and_explicit_threshold(fakes: dict[str, Any]) -> None:
    pred = TorchPredictor(Path("best.pt"), overlap=8)
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    mask = pred.predict_mask(img, img)
    assert mask.shape == (32, 32)
    assert fakes["predict_kwargs"]["threshold"] == pytest.approx(0.42)
    assert fakes["predict_kwargs"]["tile_size"] == 256
    assert fakes["predict_kwargs"]["overlap"] == 8

    pred.predict_mask(img, img, threshold=0.7)
    assert fakes["predict_kwargs"]["threshold"] == pytest.approx(0.7)
