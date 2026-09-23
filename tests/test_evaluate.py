import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from conftest import write_change_split  # noqa: E402
from geoseg.evaluation.evaluate import main, run_evaluation  # noqa: E402
from geoseg.training.config import Config  # noqa: E402
from geoseg.training.trainer import fit  # noqa: E402


@pytest.fixture
def root_with_test_split(tmp_path: Path) -> Path:
    write_change_split(tmp_path, "train", n_images=3, size=128, seed=0)
    write_change_split(tmp_path, "val", n_images=2, size=128, seed=1)
    write_change_split(tmp_path, "test", n_images=2, size=128, seed=2)
    return tmp_path


def _train_tiny_checkpoint(root: Path, output_dir: Path) -> Path:
    cfg = Config.model_validate(
        {
            "seed": 0,
            "data": {"root": str(root), "tile_size": 64, "num_workers": 0},
            "model": {"encoder": "resnet18", "encoder_weights": None},
            "train": {"epochs": 1, "batch_size": 4, "amp": False, "log_every": 0},
            "logging": {"output_dir": str(output_dir)},
        }
    )
    fit(cfg)
    return output_dir / "best.pt"


def test_run_evaluation_on_test_split(root_with_test_split: Path, tmp_path: Path) -> None:
    checkpoint = _train_tiny_checkpoint(root_with_test_split, tmp_path / "run")

    metrics = run_evaluation(checkpoint, split="test")

    assert metrics["split"] == "test"
    assert metrics["n_images"] == 2
    assert metrics["n_tiles"] == 8  # 2 images x 128x128 / 64x64 tiles = 4 tiles each
    for key in ("iou", "f1", "precision", "recall", "loss"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_run_evaluation_overrides_data_root(root_with_test_split: Path, tmp_path: Path) -> None:
    """A checkpoint's stored config.data.root should be overridable (e.g. Kaggle -> local)."""
    checkpoint = _train_tiny_checkpoint(root_with_test_split, tmp_path / "run")

    # Simulate a checkpoint trained with a different (now-invalid) data root.
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt["config"]["data"]["root"] = "/kaggle/working/levir-cd"  # would not exist here
    torch.save(ckpt, checkpoint)

    metrics = run_evaluation(checkpoint, data_root=str(root_with_test_split), split="val")
    assert metrics["n_images"] == 2  # val split has 2 images


def test_cli_writes_metrics_json(root_with_test_split: Path, tmp_path: Path) -> None:
    checkpoint = _train_tiny_checkpoint(root_with_test_split, tmp_path / "run")
    output = tmp_path / "test_metrics.json"

    exit_code = main(["--checkpoint", str(checkpoint), "--split", "test", "--output", str(output)])

    assert exit_code == 0
    saved = json.loads(output.read_text())
    assert saved["split"] == "test"
    assert "f1" in saved
