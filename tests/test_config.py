from pathlib import Path

import pytest
from pydantic import ValidationError

from geoseg.training.config import Config, apply_overrides, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.seed == 42
    assert cfg.data.tile_size == 256
    assert cfg.train.loss == "dice_focal"
    assert cfg.model.fusion == "diff"


def test_yaml_defaults_match_code_defaults() -> None:
    assert load_config(DEFAULT_CONFIG) == Config()


def test_overrides_are_applied_and_typed() -> None:
    cfg = load_config(
        DEFAULT_CONFIG,
        ["train.epochs=3", "train.lr=3e-4", "model.encoder_weights=null", "train.amp=false"],
    )
    assert cfg.train.epochs == 3
    assert cfg.train.lr == pytest.approx(3e-4)
    assert cfg.model.encoder_weights is None
    assert cfg.train.amp is False


def test_apply_overrides_does_not_mutate_input() -> None:
    raw = {"train": {"epochs": 1}}
    out = apply_overrides(raw, ["train.epochs=5"])
    assert raw["train"]["epochs"] == 1
    assert out["train"]["epochs"] == 5


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, ["train.epoch=3"])  # typo: epoch vs epochs


def test_invalid_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, ["train.loss=mse"])
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, ["train.batch_size=0"])


def test_malformed_override_raises() -> None:
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["train.epochs"])
