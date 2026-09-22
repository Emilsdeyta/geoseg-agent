import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from geoseg.models.siamese_unet import SiameseUNet  # noqa: E402
from geoseg.training.config import Config  # noqa: E402
from geoseg.training.losses import build_loss  # noqa: E402
from geoseg.training.trainer import build_model, fit  # noqa: E402


def test_model_can_overfit_one_batch() -> None:
    """If this fails, something in forward/loss/optimizer wiring is broken."""
    torch.manual_seed(0)
    image_a = torch.randn(2, 3, 64, 64)
    image_b = image_a.clone()
    image_b[:, :, 16:48, 16:48] += 3.0  # the "change"
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1.0

    model = SiameseUNet("resnet18", None, decoder_channels=(64, 32, 16, 8, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = build_loss("dice_focal")

    model.train()
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = loss_fn(model(image_a, image_b), mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def _tiny_config(root: Path, output_dir: Path, epochs: int, patience: int = 0) -> Config:
    return Config.model_validate(
        {
            "seed": 0,
            "data": {"root": str(root), "tile_size": 64, "num_workers": 0},
            "model": {"encoder": "resnet18", "encoder_weights": None},
            "train": {
                "epochs": epochs,
                "batch_size": 4,
                "lr": 1e-3,
                "amp": False,
                "patience": patience,
                "log_every": 0,
            },
            "logging": {"output_dir": str(output_dir)},
        }
    )


def test_fit_end_to_end_writes_artifacts(synthetic_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    summary = fit(_tiny_config(synthetic_root, out, epochs=2))

    assert summary["epochs_run"] == 2
    assert summary["best_epoch"] in (1, 2)
    for key in ("iou", "f1", "precision", "recall", "loss"):
        assert key in summary["best_val"]
    for key in ("iou", "f1", "precision", "recall"):
        assert 0.0 <= summary["best_val"][key] <= 1.0
    for name in ("config.yaml", "history.json", "last.pt", "best.pt"):
        assert (out / name).is_file()


def test_checkpoint_can_rebuild_the_model(synthetic_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    fit(_tiny_config(synthetic_root, out, epochs=1))

    checkpoint = torch.load(out / "best.pt", map_location="cpu")
    rebuilt = build_model(Config.model_validate(checkpoint["config"]))
    rebuilt.load_state_dict(checkpoint["model"])  # same architecture -> loads cleanly
    assert checkpoint["epoch"] == 1


def test_limit_batches_and_early_stopping(synthetic_root: Path, tmp_path: Path) -> None:
    cfg = _tiny_config(synthetic_root, tmp_path / "run", epochs=20, patience=1)
    cfg.train.limit_batches = 1
    summary = fit(cfg)
    assert summary["epochs_run"] < 20  # patience=1 stops long before 20 epochs


def test_resume_auto_continues_history_and_epochs(synthetic_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    first = fit(_tiny_config(synthetic_root, out, epochs=2))
    assert first["epochs_run"] == 2

    cfg = _tiny_config(synthetic_root, out, epochs=4)
    second = fit(cfg, resume="auto")

    assert second["epochs_run"] == 4  # 2 already done + 2 more, not restarted from 0
    history = json.loads((out / "history.json").read_text())
    assert [row["epoch"] for row in history] == [1.0, 2.0, 3.0, 4.0]


def test_resume_auto_without_checkpoint_starts_fresh(synthetic_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"  # nothing written here yet
    summary = fit(_tiny_config(synthetic_root, out, epochs=1), resume="auto")
    assert summary["epochs_run"] == 1


def test_resume_explicit_checkpoint_path(synthetic_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    fit(_tiny_config(synthetic_root, out, epochs=1))
    checkpoint = out / "last.pt"

    resumed_out = tmp_path / "resumed"
    resumed_out.mkdir()
    # Continuing into a *different* output_dir with an explicit checkpoint path still
    # picks up epoch/optimizer/scaler state, it just has no prior history.json to extend.
    cfg = _tiny_config(synthetic_root, resumed_out, epochs=2)
    summary = fit(cfg, resume=str(checkpoint))

    history = json.loads((resumed_out / "history.json").read_text())
    assert [row["epoch"] for row in history] == [2.0]  # only epoch 2 ran here
    assert summary["epochs_run"] == 1


def test_resume_missing_explicit_checkpoint_raises(synthetic_root: Path, tmp_path: Path) -> None:
    cfg = _tiny_config(synthetic_root, tmp_path / "run", epochs=1)
    with pytest.raises(FileNotFoundError):
        fit(cfg, resume=str(tmp_path / "nope.pt"))


def test_cli_resume_flag(synthetic_root: Path, tmp_path: Path) -> None:
    from geoseg.training.train import main

    default_config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    out = tmp_path / "cli_run"
    base_args = [
        "--config",
        str(default_config),
        "--set",
        f"data.root={synthetic_root}",
        "data.tile_size=64",
        "data.num_workers=0",
        "model.encoder=resnet18",
        "model.encoder_weights=null",
        "train.batch_size=4",
        "train.amp=false",
        f"logging.output_dir={out}",
    ]
    assert main([*base_args, "train.epochs=1"]) == 0
    assert main([*base_args, "train.epochs=2", "--resume", "auto"]) == 0
    history = json.loads((out / "history.json").read_text())
    assert [row["epoch"] for row in history] == [1.0, 2.0]


def test_cli_runs_with_overrides(synthetic_root: Path, tmp_path: Path) -> None:
    from geoseg.training.train import main

    default_config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    out = tmp_path / "cli_run"
    exit_code = main(
        [
            "--config",
            str(default_config),
            "--set",
            f"data.root={synthetic_root}",
            "data.tile_size=64",
            "data.num_workers=0",
            "model.encoder=resnet18",
            "model.encoder_weights=null",
            "train.epochs=1",
            "train.batch_size=4",
            "train.amp=false",
            f"logging.output_dir={out}",
        ]
    )
    assert exit_code == 0
    assert (out / "best.pt").is_file()
