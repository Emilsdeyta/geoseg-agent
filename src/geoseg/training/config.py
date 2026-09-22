"""Typed, validated training configuration (no torch dependency).

The YAML file is the source of truth; every value can be overridden from the CLI
with ``--set section.key=value`` (handy on Kaggle where paths differ).
Unknown keys are rejected so a typo in a config never silently does nothing.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Section):
    root: str = "data/raw/levir-cd"
    tile_size: int = Field(default=256, gt=0)
    overlap: int = Field(default=0, ge=0)
    num_workers: int = Field(default=2, ge=0)
    cache: bool = False  # keep decoded images in RAM: much faster, ~3.5 GB for LEVIR-CD


class ModelConfig(_Section):
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    fusion: Literal["diff", "concat"] = "diff"


class TrainConfig(_Section):
    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=8, gt=0)
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    amp: bool = True
    loss: Literal["bce", "dice", "focal", "dice_focal"] = "dice_focal"
    focal_gamma: float = Field(default=2.0, ge=0)
    focal_alpha: float | None = Field(default=None, ge=0, le=1)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    patience: int = Field(default=10, ge=0)  # early stopping on val F1; 0 disables it
    log_every: int = Field(default=100, ge=0)  # log every N train steps; 0 disables it
    limit_batches: int = Field(default=0, ge=0)  # debug: only N batches per epoch; 0 = all


class LoggingConfig(_Section):
    output_dir: str = "outputs/baseline"
    tracker: Literal["none", "mlflow"] = "none"
    experiment: str = "geoseg-levir-cd"


class Config(_Section):
    seed: int = 42
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    logging: LoggingConfig = LoggingConfig()


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``"a.b=value"`` overrides to a nested dict; values are parsed as YAML."""
    result = copy.deepcopy(raw)
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Override must look like 'section.key=value', got {item!r}")
        *parents, leaf = key.split(".")
        node = result
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override {key!r}: {part!r} is not a section")
            node = child
        node[leaf] = yaml.safe_load(value)
    return result


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Load a YAML config, apply CLI overrides and validate."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config.model_validate(apply_overrides(raw, overrides))
