import pytest

torch = pytest.importorskip("torch")

from geoseg.training.losses import (  # noqa: E402
    DiceFocalLoss,
    DiceLoss,
    FocalLoss,
    build_loss,
)


def _target() -> "torch.Tensor":
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1.0
    return target


def _confident_logits(target: "torch.Tensor", value: float = 12.0) -> "torch.Tensor":
    return (target * 2 - 1) * value  # +value on positives, -value on negatives


def test_dice_is_zero_for_perfect_and_near_one_for_inverted_prediction() -> None:
    target = _target()
    loss = DiceLoss()
    assert loss(_confident_logits(target), target).item() < 1e-2
    assert loss(-_confident_logits(target), target).item() > 0.9


def test_focal_with_gamma_zero_equals_bce() -> None:
    logits = torch.randn(2, 1, 8, 8)
    target = _target()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert FocalLoss(gamma=0.0)(logits, target).item() == pytest.approx(bce.item(), rel=1e-5)


def test_focal_downweights_easy_pixels() -> None:
    logits = torch.randn(2, 1, 8, 8)
    target = _target()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert FocalLoss(gamma=2.0)(logits, target).item() < bce.item()


def test_focal_alpha_weights_positive_class() -> None:
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    plain = FocalLoss(gamma=0.0)(logits, target).item()
    weighted = FocalLoss(gamma=0.0, alpha=0.25)(logits, target).item()
    assert weighted == pytest.approx(0.25 * plain, rel=1e-5)


def test_dice_focal_is_weighted_sum() -> None:
    logits = torch.randn(2, 1, 8, 8)
    target = _target()
    combined = DiceFocalLoss(dice_weight=1.0, focal_weight=0.5)(logits, target)
    expected = DiceLoss()(logits, target) + 0.5 * FocalLoss()(logits, target)
    assert combined.item() == pytest.approx(expected.item(), rel=1e-5)


def test_losses_are_differentiable() -> None:
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    for name in ("bce", "dice", "focal", "dice_focal"):
        logits.grad = None
        build_loss(name)(logits, _target()).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def test_build_loss_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("mse")
