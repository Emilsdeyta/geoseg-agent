import numpy as np
import pytest

from geoseg.evaluation.comparison import compose_comparison_image, rank_records


def test_rank_records_returns_worst_then_best() -> None:
    ious = [0.9, 0.1, 0.5, 0.7, 0.3]
    records = [{"name": f"img{i}.png", "iou": iou} for i, iou in enumerate(ious)]
    best, worst = rank_records(records, n_best=2, n_worst=2)
    assert [r["name"] for r in worst] == ["img1.png", "img4.png"]  # iou 0.1, 0.3
    assert [r["name"] for r in best] == ["img3.png", "img0.png"]  # iou 0.7, 0.9


def test_rank_records_n_best_zero_returns_empty_best() -> None:
    records = [{"name": "a", "iou": 0.5}, {"name": "b", "iou": 0.9}]
    best, worst = rank_records(records, n_best=0, n_worst=1)
    assert best == []
    assert [r["name"] for r in worst] == ["a"]


def test_rank_records_handles_fewer_records_than_requested() -> None:
    records = [{"name": "a", "iou": 0.5}]
    best, worst = rank_records(records, n_best=5, n_worst=5)
    assert len(best) == 1
    assert len(worst) == 1


def test_compose_comparison_image_has_expected_size() -> None:
    size = 8
    image_a = np.zeros((size, size, 3), dtype=np.uint8)
    image_b = np.zeros((size, size, 3), dtype=np.uint8)
    gt_mask = np.zeros((size, size), dtype=np.uint8)
    pred_mask = np.zeros((size, size), dtype=np.uint8)

    composite = compose_comparison_image(
        image_a, image_b, gt_mask, pred_mask, title="test", panel_size=size
    )

    gap = 6
    n_panels = 5
    assert composite.size == (size * n_panels + gap * (n_panels - 1), size + 28)


@pytest.mark.parametrize(
    ("pred", "gt", "expected_rgb"),
    [
        (1, 1, (0, 200, 0)),  # true positive -> green
        (1, 0, (220, 0, 0)),  # false positive -> red
        (0, 1, (0, 80, 220)),  # false negative -> blue
        (0, 0, (0, 0, 0)),  # true negative -> black
    ],
)
def test_compose_comparison_image_error_colors(
    pred: int, gt: int, expected_rgb: tuple[int, int, int]
) -> None:
    size = 4
    image_a = np.zeros((size, size, 3), dtype=np.uint8)
    image_b = np.zeros((size, size, 3), dtype=np.uint8)
    gt_mask = np.full((size, size), gt, dtype=np.uint8)
    pred_mask = np.full((size, size), pred, dtype=np.uint8)

    composite = compose_comparison_image(
        image_a, image_b, gt_mask, pred_mask, title="t", panel_size=size
    )

    # Error map is the 5th (last) panel; sample its centre pixel, below the label bar.
    error_panel_x0 = 4 * (size + 6)
    pixel = composite.getpixel((error_panel_x0 + size // 2, 24 + size - 1))
    assert pixel == expected_rgb
