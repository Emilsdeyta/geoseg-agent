from pathlib import Path

import pytest

from geoseg.training.config import Config
from geoseg.training.tracking import NullTracker, flatten, make_tracker


def test_flatten_nested_mapping() -> None:
    assert flatten({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_default_tracker_is_a_noop(tmp_path: Path) -> None:
    tracker = make_tracker(Config(), tmp_path)
    assert type(tracker) is NullTracker
    tracker.log_params({"x": 1})
    tracker.log_metrics({"loss": 1.0}, step=1)
    tracker.close()


@pytest.mark.slow
def test_mlflow_tracker_logs_metrics(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from geoseg.training.tracking import MlflowTracker

    tracker = MlflowTracker("pytest-exp", tmp_path / "mlflow")
    tracker.log_params(flatten(Config().model_dump()))
    tracker.log_metrics({"val_f1": 0.25}, step=1)
    tracker.log_metrics({"val_f1": 0.75}, step=2)
    tracker.close()

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').as_posix()}")
    runs = mlflow.search_runs(experiment_names=["pytest-exp"])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.val_f1"] == 0.75
    assert runs.iloc[0]["params.train.loss"] == "dice_focal"
