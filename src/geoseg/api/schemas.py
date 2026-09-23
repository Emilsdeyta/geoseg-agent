"""Pydantic request/response models for the GeoSeg-Agent HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Always 'ok' if the process is up.")
    model_loaded: bool = Field(description="True once a checkpoint has been loaded.")
    checkpoint_epoch: int | None = Field(
        default=None, description="Epoch of the loaded checkpoint, if known."
    )


class PredictResponse(BaseModel):
    geojson: dict[str, Any] = Field(
        description="FeatureCollection of change polygons in local pixel coordinates."
    )
    report: str = Field(description="Rule-based natural-language summary of the result.")
    n_polygons: int
    changed_area_px: float
    threshold: float = Field(description="Probability threshold actually used.")
