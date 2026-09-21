"""Experiment tracking. ``NullTracker`` does nothing; ``MlflowTracker`` logs to MLflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geoseg.training.config import Config


class NullTracker:
    """Tracker that discards everything (default)."""

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass

    def close(self) -> None:
        pass


class MlflowTracker(NullTracker):
    """Logs params, per-epoch metrics and artifacts to a local MLflow store."""

    def __init__(self, experiment: str, tracking_dir: Path) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("tracker=mlflow needs MLflow: pip install mlflow") from exc
        self._mlflow = mlflow
        tracking_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / 'mlflow.db').resolve().as_posix()}")
        mlflow.set_experiment(experiment)
        mlflow.start_run()

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def close(self) -> None:
        self._mlflow.end_run()


def flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"train": {"lr": 1}}`` -> ``{"train.lr": 1}`` (MLflow params are flat)."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def make_tracker(cfg: Config, output_dir: Path) -> NullTracker:
    if cfg.logging.tracker == "mlflow":
        return MlflowTracker(cfg.logging.experiment, output_dir / "mlflow")
    return NullTracker()
