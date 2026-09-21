import pytest

torch = pytest.importorskip("torch")

from geoseg.training.metrics import BinaryConfusion  # noqa: E402


def _batch() -> "tuple[torch.Tensor, torch.Tensor]":
    # 1 x 1 x 2 x 4 -> 8 pixels: tp=2, fp=1, fn=2, tn=3
    target = torch.tensor([[[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]])
    logits = torch.tensor([[[[5.0, 5.0, -5.0, -5.0], [5.0, -5.0, -5.0, -5.0]]]])
    return logits, target


def test_counts_and_metrics_on_known_example() -> None:
    logits, target = _batch()
    confusion = BinaryConfusion()
    confusion.update(logits, target)
    assert (confusion.tp, confusion.fp, confusion.fn, confusion.tn) == (2, 1, 2, 3)
    metrics = confusion.compute()
    assert metrics["iou"] == pytest.approx(2 / 5)
    assert metrics["f1"] == pytest.approx(4 / 7)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 4)


def test_accumulation_over_batches_equals_single_pass() -> None:
    logits, target = _batch()
    once = BinaryConfusion()
    once.update(logits.repeat(2, 1, 1, 1), target.repeat(2, 1, 1, 1))
    twice = BinaryConfusion()
    twice.update(logits, target)
    twice.update(logits, target)
    assert once.compute() == twice.compute()


def test_threshold_changes_predictions() -> None:
    logits = torch.zeros(1, 1, 2, 2)  # probability 0.5 everywhere
    target = torch.ones(1, 1, 2, 2)
    low, high = BinaryConfusion(), BinaryConfusion()
    low.update(logits, target, threshold=0.4)
    high.update(logits, target, threshold=0.6)
    assert low.tp == 4
    assert high.tp == 0


def test_empty_counts_give_zero_not_error() -> None:
    assert BinaryConfusion().compute() == {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}


def test_reset() -> None:
    logits, target = _batch()
    confusion = BinaryConfusion()
    confusion.update(logits, target)
    confusion.reset()
    assert (confusion.tp, confusion.fp, confusion.fn, confusion.tn) == (0, 0, 0, 0)
