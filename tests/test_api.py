"""Tests for the FastAPI layer. Torch-free: a fake predictor stands in for the model."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("shapely")
pytest.importorskip("skimage")
try:  # FastAPI needs python-multipart for File/Form; module name differs by version.
    import python_multipart  # noqa: F401
except ImportError:
    pytest.importorskip("multipart")

from fastapi.testclient import TestClient
from PIL import Image

from geoseg.api import main as api_main
from geoseg.api.main import create_app


class FakePredictor:
    checkpoint_epoch: int | None = 42
    default_threshold: float = 0.5

    def __init__(self, mask: np.ndarray | None = None) -> None:
        self.mask = mask
        self.calls: list[dict[str, Any]] = []

    def predict_mask(
        self, image_a: np.ndarray, image_b: np.ndarray, *, threshold: float | None = None
    ) -> np.ndarray:
        self.calls.append({"shape": image_a.shape, "threshold": threshold})
        if self.mask is not None:
            return self.mask
        mask = np.zeros(image_a.shape[:2], dtype=bool)
        mask[5:15, 5:15] = True  # 100 px^2
        mask[30:40, 30:50] = True  # 200 px^2
        return mask


def _png(size: tuple[int, int] = (64, 64), color: int = 0) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (color, color, color)).save(buf, format="PNG")
    return buf.getvalue()


def _files(a: bytes | None = None, b: bytes | None = None) -> dict[str, tuple[str, bytes, str]]:
    return {
        "image_a": ("before.png", a if a is not None else _png(), "image/png"),
        "image_b": ("after.png", b if b is not None else _png(color=255), "image/png"),
    }


@pytest.fixture
def fake() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(fake: FakePredictor) -> TestClient:
    return TestClient(create_app(predictor=fake))


def test_health_without_model() -> None:
    body = TestClient(create_app()).get("/health").json()
    assert body == {"status": "ok", "model_loaded": False, "checkpoint_epoch": None}


def test_health_with_model(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["checkpoint_epoch"] == 42


def test_predict_returns_503_without_model() -> None:
    resp = TestClient(create_app()).post("/predict", files=_files())
    assert resp.status_code == 503


def test_predict_happy_path(client: TestClient) -> None:
    resp = client.post("/predict", files=_files())
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_polygons"] == 2
    assert body["changed_area_px"] == pytest.approx(300.0, rel=0.02)
    assert body["threshold"] == 0.5
    assert body["geojson"]["type"] == "FeatureCollection"
    assert len(body["geojson"]["features"]) == 2
    assert "2 change region" in body["report"]
    assert "before.png" in body["report"]


def test_predict_forwards_threshold(client: TestClient, fake: FakePredictor) -> None:
    resp = client.post("/predict", files=_files(), data={"threshold": "0.3"})
    assert resp.status_code == 200
    assert resp.json()["threshold"] == pytest.approx(0.3)
    assert fake.calls[-1]["threshold"] == pytest.approx(0.3)


def test_predict_uses_default_threshold_when_omitted(
    client: TestClient, fake: FakePredictor
) -> None:
    client.post("/predict", files=_files())
    assert fake.calls[-1]["threshold"] == pytest.approx(0.5)


def test_predict_min_area_filters_small_regions(client: TestClient) -> None:
    resp = client.post("/predict", files=_files(), data={"min_area": "150"})
    assert resp.status_code == 200
    assert resp.json()["n_polygons"] == 1


def test_predict_empty_mask(fake: FakePredictor) -> None:
    empty = FakePredictor(mask=np.zeros((64, 64), dtype=bool))
    resp = TestClient(create_app(predictor=empty)).post("/predict", files=_files())
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_polygons"] == 0
    assert body["geojson"]["features"] == []


@pytest.mark.parametrize("bad", ["-0.1", "1.5"])
def test_predict_rejects_out_of_range_threshold(client: TestClient, bad: str) -> None:
    resp = client.post("/predict", files=_files(), data={"threshold": bad})
    assert resp.status_code == 422


def test_predict_rejects_negative_min_area(client: TestClient) -> None:
    resp = client.post("/predict", files=_files(), data={"min_area": "-1"})
    assert resp.status_code == 422


def test_predict_rejects_non_image(client: TestClient) -> None:
    resp = client.post("/predict", files=_files(a=b"definitely not an image"))
    assert resp.status_code == 400
    assert "image_a" in resp.json()["detail"]


def test_predict_rejects_size_mismatch(client: TestClient) -> None:
    resp = client.post("/predict", files=_files(a=_png((64, 64)), b=_png((32, 32))))
    assert resp.status_code == 400
    assert "differ" in resp.json()["detail"]


def test_predict_rejects_missing_file(client: TestClient) -> None:
    resp = client.post("/predict", files={"image_a": ("a.png", _png(), "image/png")})
    assert resp.status_code == 422


def test_predict_rejects_oversized_upload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 10)
    resp = client.post("/predict", files=_files())
    assert resp.status_code == 413


def test_metrics_are_forwarded_to_report(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_report(geojson: dict[str, Any], **kwargs: Any) -> str:
        seen.update(kwargs)
        return "stub report"

    monkeypatch.setattr(api_main, "generate_report", fake_report)
    metrics = {"iou": 0.82, "f1": 0.90}
    app = create_app(predictor=FakePredictor(), metrics=metrics)
    resp = TestClient(app).post("/predict", files=_files())
    assert resp.json()["report"] == "stub report"
    assert seen["metrics"] == metrics
    assert seen["image_name"] == "before.png"


def test_load_metrics_handles_missing_and_bad_files(tmp_path: Any) -> None:
    assert api_main._load_metrics(None) is None
    assert api_main._load_metrics(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert api_main._load_metrics(bad) is None
    good = tmp_path / "ok.json"
    good.write_text('{"iou": 0.8}')
    assert api_main._load_metrics(good) == {"iou": 0.8}


def test_lifespan_loads_metrics_from_env(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tmp_path / "m.json"
    m.write_text('{"iou": 0.8}')
    monkeypatch.setenv(api_main.METRICS_ENV, str(m))
    monkeypatch.delenv(api_main.CHECKPOINT_ENV, raising=False)
    app = create_app()
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert app.state.metrics == {"iou": 0.8}
