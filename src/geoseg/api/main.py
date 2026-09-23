"""FastAPI application for GeoSeg-Agent.

Run locally::

    set GEOSEG_CHECKPOINT=outputs\\full_run\\best.pt
    set GEOSEG_METRICS=outputs\\full_run\\test_metrics.json   (optional)
    uvicorn geoseg.api.main:app --port 8000

Endpoints:

* ``GET /health``  - liveness + whether a model is loaded.
* ``POST /predict`` - multipart upload of ``image_a`` (before) and ``image_b``
  (after); returns change polygons (GeoJSON, local pixel coordinates) and a
  rule-based natural-language report.

The checkpoint is loaded once at startup, not per request. If
``GEOSEG_CHECKPOINT`` is unset the server still starts (``/health`` works) and
``/predict`` answers 503.
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from geoseg.agent.report import generate_report
from geoseg.api.predictor import ChangePredictor
from geoseg.api.schemas import HealthResponse, PredictResponse
from geoseg.inference.polygonize import mask_to_polygons, polygons_to_geojson

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # per image
CHECKPOINT_ENV = "GEOSEG_CHECKPOINT"
METRICS_ENV = "GEOSEG_METRICS"


def _read_rgb(upload: UploadFile, field: str) -> np.ndarray:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{field} exceeds {MAX_UPLOAD_BYTES} bytes.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail=f"{field} is not a readable image.") from exc


def _load_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning("Could not read metrics file %s; reports will omit metrics.", path)
        return None
    return loaded if isinstance(loaded, dict) else None


def create_app(
    predictor: ChangePredictor | None = None,
    metrics: dict[str, Any] | None = None,
) -> FastAPI:
    """Build the app. Tests inject a fake ``predictor``; production loads from env."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.predictor is None:
            ckpt = os.environ.get(CHECKPOINT_ENV)
            if ckpt:
                from geoseg.api.predictor import TorchPredictor

                app.state.predictor = TorchPredictor(Path(ckpt))
            else:
                logger.warning("%s is not set; /predict will return 503.", CHECKPOINT_ENV)
        if app.state.metrics is None:
            m_path = os.environ.get(METRICS_ENV)
            app.state.metrics = _load_metrics(Path(m_path) if m_path else None)
        yield

    app = FastAPI(title="GeoSeg-Agent", version="0.1.0", lifespan=lifespan)
    app.state.predictor = predictor
    app.state.metrics = metrics

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        pred: ChangePredictor | None = app.state.predictor
        return HealthResponse(
            status="ok",
            model_loaded=pred is not None,
            checkpoint_epoch=pred.checkpoint_epoch if pred is not None else None,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        image_a: Annotated[UploadFile, File(description="'Before' image.")],
        image_b: Annotated[UploadFile, File(description="'After' image.")],
        threshold: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
        min_area: Annotated[float, Form(ge=0.0)] = 0.0,
    ) -> PredictResponse:
        pred: ChangePredictor | None = app.state.predictor
        if pred is None:
            raise HTTPException(status_code=503, detail="No model loaded.")

        arr_a = _read_rgb(image_a, "image_a")
        arr_b = _read_rgb(image_b, "image_b")
        if arr_a.shape != arr_b.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Image sizes differ: {arr_a.shape[:2]} vs {arr_b.shape[:2]}.",
            )

        used_threshold = pred.default_threshold if threshold is None else threshold
        mask = pred.predict_mask(arr_a, arr_b, threshold=used_threshold)
        polygons = mask_to_polygons(mask, min_area=min_area)
        name = image_a.filename or "image_a"
        geojson = polygons_to_geojson(
            polygons,
            image_name=name,
            extra_properties={
                "checkpoint_epoch": pred.checkpoint_epoch,
                "threshold": used_threshold,
            },
        )
        report = generate_report(geojson, metrics=app.state.metrics, image_name=name)
        return PredictResponse(
            geojson=geojson,
            report=report,
            n_polygons=len(polygons),
            changed_area_px=float(sum(p.area for p in polygons)),
            threshold=used_threshold,
        )

    return app


app = create_app()
