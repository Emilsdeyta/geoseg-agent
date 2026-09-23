from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from conftest import write_change_split  # noqa: E402
from geoseg.evaluation.visualize import export_error_analysis, main  # noqa: E402
from geoseg.training.config import Config  # noqa: E402
from geoseg.training.trainer import fit  # noqa: E402


@pytest.fixture
def root_with_test_split(tmp_path: Path) -> Path:
    write_change_split(tmp_path, "train", n_images=3, size=128, seed=0)
    write_change_split(tmp_path, "val", n_images=2, size=128, seed=1)
    write_change_split(tmp_path, "test", n_images=4, size=128, seed=2)
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


def test_export_error_analysis_writes_pngs_and_ranking(
    root_with_test_split: Path, tmp_path: Path
) -> None:
    checkpoint = _train_tiny_checkpoint(root_with_test_split, tmp_path / "run")
    out_dir = tmp_path / "error_analysis"

    summary = export_error_analysis(
        checkpoint,
        data_root=None,
        split="test",
        output_dir=out_dir,
        n_best=2,
        n_worst=2,
        threshold=None,
    )

    assert summary["n_images"] == 4
    assert summary["exported_best"] == 2
    assert summary["exported_worst"] == 2
    assert (out_dir / "ranking.json").is_file()
    png_files = sorted(out_dir.glob("*.png"))
    assert len(png_files) == 4  # 2 best + 2 worst, no overlap since 2+2 == n_images


def test_export_error_analysis_caps_at_available_images(
    root_with_test_split: Path, tmp_path: Path
) -> None:
    """Asking for more best/worst images than exist should not error or duplicate."""
    checkpoint = _train_tiny_checkpoint(root_with_test_split, tmp_path / "run")
    out_dir = tmp_path / "error_analysis"

    summary = export_error_analysis(
        checkpoint,
        data_root=None,
        split="test",
        output_dir=out_dir,
        n_best=10,
        n_worst=10,
        threshold=None,
    )
    assert summary["exported_best"] == 4
    assert summary["exported_worst"] == 4


def test_cli_runs_end_to_end(root_with_test_split: Path, tmp_path: Path) -> None:
    checkpoint = _train_tiny_checkpoint(root_with_test_split, tmp_path / "run")
    out_dir = tmp_path / "cli_error_analysis"

    exit_code = main(
        [
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--output-dir",
            str(out_dir),
            "--n-best",
            "1",
            "--n-worst",
            "1",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "ranking.json").is_file()
