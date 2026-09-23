from __future__ import annotations

import numpy as np
import pytest

from geoseg.agent.report import generate_report, summarize_geojson
from geoseg.inference.polygonize import mask_to_polygons, polygons_to_geojson


def _fake_geojson(areas: list[float], image_name: str | None = "test_1.png") -> dict:
    """Build a minimal fake GeoJSON with the given feature areas, without
    needing real polygon geometry — report.py only reads properties."""
    features = []
    for area in areas:
        props: dict = {"area_px": area}
        if image_name is not None:
            props["image_name"] = image_name
        features.append({"type": "Feature", "geometry": None, "properties": props})
    return {"type": "FeatureCollection", "features": features}


# --- summarize_geojson --------------------------------------------------


def test_summarize_empty_geojson() -> None:
    summary = summarize_geojson(_fake_geojson([]))
    assert summary["n_regions"] == 0
    assert summary["total_area_px"] == 0.0
    assert summary["largest_area_px"] == 0.0
    assert summary["smallest_area_px"] == 0.0
    assert summary["mean_area_px"] == 0.0
    assert summary["largest_region_size_class"] is None


def test_summarize_single_region() -> None:
    summary = summarize_geojson(_fake_geojson([250.0]))
    assert summary["n_regions"] == 1
    assert summary["total_area_px"] == 250.0
    assert summary["largest_area_px"] == 250.0
    assert summary["smallest_area_px"] == 250.0
    assert summary["mean_area_px"] == 250.0
    assert summary["largest_region_size_class"] == "medium"


def test_summarize_multiple_regions() -> None:
    summary = summarize_geojson(_fake_geojson([10.0, 500.0, 2000.0]))
    assert summary["n_regions"] == 3
    assert summary["total_area_px"] == pytest.approx(2510.0)
    assert summary["largest_area_px"] == 2000.0
    assert summary["smallest_area_px"] == 10.0
    assert summary["mean_area_px"] == pytest.approx(2510.0 / 3)
    assert summary["largest_region_size_class"] == "large"


def test_summarize_picks_up_image_name() -> None:
    summary = summarize_geojson(_fake_geojson([50.0], image_name="before_after.png"))
    assert summary["image_name"] == "before_after.png"


def test_summarize_missing_image_name_is_none() -> None:
    summary = summarize_geojson(_fake_geojson([50.0], image_name=None))
    assert summary["image_name"] is None


@pytest.mark.parametrize(
    ("area", "expected_class"),
    [(5.0, "small"), (99.9, "small"), (500.0, "medium"), (1000.1, "large"), (5000.0, "large")],
)
def test_size_classification_thresholds(area: float, expected_class: str) -> None:
    summary = summarize_geojson(_fake_geojson([area]))
    assert summary["largest_region_size_class"] == expected_class


# --- generate_report ------------------------------------------------------


def test_report_no_regions_mentions_no_changes() -> None:
    report = generate_report(_fake_geojson([]))
    assert "no significant changes" in report.lower()


def test_report_single_small_region() -> None:
    report = generate_report(_fake_geojson([20.0]))
    assert "1 change region" in report
    assert "small" in report


def test_report_single_large_region() -> None:
    report = generate_report(_fake_geojson([5000.0]))
    assert "large" in report


def test_report_multiple_regions_uses_plural_and_range() -> None:
    report = generate_report(_fake_geojson([10.0, 500.0, 2000.0]))
    assert "3 change regions" in report
    assert "On average" in report
    assert "10" in report  # smallest, appears in the range sentence
    assert "2000" in report  # largest


def test_report_without_metrics_omits_iou_sentence() -> None:
    report = generate_report(_fake_geojson([100.0]), metrics=None)
    assert "IoU" not in report


def test_report_with_metrics_mentions_iou_and_f1() -> None:
    report = generate_report(_fake_geojson([100.0]), metrics={"iou": 0.8183, "f1": 0.9001})
    assert "IoU of 0.82" in report
    assert "F1 of 0.90" in report


def test_report_with_partial_metrics_is_skipped() -> None:
    """If metrics is given but missing one of iou/f1, don't guess — omit
    the sentence entirely rather than reporting a half-true number."""
    report = generate_report(_fake_geojson([100.0]), metrics={"iou": 0.8})
    assert "IoU" not in report


def test_report_caveat_included_by_default() -> None:
    report = generate_report(_fake_geojson([100.0]))
    assert "Note on reliability" in report


def test_report_caveat_can_be_disabled() -> None:
    report = generate_report(_fake_geojson([100.0]), include_caveat=False)
    assert "Note on reliability" not in report


def test_report_image_name_override_wins_over_geojson() -> None:
    report = generate_report(
        _fake_geojson([10.0], image_name="from_geojson.png"), image_name="override.png"
    )
    assert "override.png" in report
    assert "from_geojson.png" not in report


def test_report_falls_back_to_geojson_image_name() -> None:
    report = generate_report(_fake_geojson([10.0], image_name="from_geojson.png"))
    assert "from_geojson.png" in report


def test_report_generic_subject_when_no_name_available() -> None:
    report = generate_report(_fake_geojson([10.0], image_name=None))
    assert "In this image pair" in report


def test_report_is_deterministic() -> None:
    geojson = _fake_geojson([10.0, 500.0])
    metrics = {"iou": 0.8, "f1": 0.9}
    assert generate_report(geojson, metrics) == generate_report(geojson, metrics)


# --- integration with the real polygonize.py output ------------------------


def test_report_end_to_end_from_real_polygonize_output() -> None:
    """Exercises summarize_geojson/generate_report against a GeoJSON built
    by the actual polygonize.py, not a hand-built fake, to catch any
    mismatch between the two modules' assumptions about feature shape."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # ~100 px^2, "medium"-ish
    mask[50:90, 50:90] = 1  # ~1600 px^2, "large"

    polygons = mask_to_polygons(mask)
    geojson = polygons_to_geojson(polygons, image_name="integration_test.png")

    report = generate_report(geojson, metrics={"iou": 0.82, "f1": 0.90})

    assert "integration_test.png" in report
    assert "2 change regions" in report
    assert "large" in report
    assert "IoU of 0.82" in report
    assert "Note on reliability" in report
