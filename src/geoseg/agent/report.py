"""Turn a change-detection GeoJSON (and, optionally, test-set metrics) into
a short, human-readable natural-language report — no LLM, no API key, no
network call.

This is deliberately built in two layers:

1. ``summarize_geojson`` extracts plain structured facts (region count,
   areas, size classification) from a GeoJSON produced by
   ``geoseg.inference.polygonize``. This is pure data extraction with no
   wording decisions in it.
2. ``generate_report`` turns that structured summary into a short
   paragraph of natural-language text, using fixed templates.

The split matters for where this project is headed next: if an LLM-backed
agent is added later, it can call ``summarize_geojson`` as one of its
"tools" and write its own prose around the same facts, instead of
duplicating the extraction logic. ``generate_report`` is the free,
zero-dependency fallback that works with no API key at all, and can keep
running as a baseline/offline mode even after an LLM layer exists.

Region-size thresholds (``SMALL_AREA_PX``, ``LARGE_AREA_PX``) are a
readability heuristic for grouping regions into "small/medium/large" in
the generated text, not a measured or scientifically derived boundary —
this is called out explicitly here and in the report text itself so it is
never mistaken for a validated claim.
"""

from __future__ import annotations

from typing import Any

__all__ = ["LARGE_AREA_PX", "SMALL_AREA_PX", "generate_report", "summarize_geojson"]

# Heuristic-only size buckets, in pixel^2, used purely to make the generated
# text more readable (e.g. "a large region of about 1,200 px^2"). These are
# NOT derived from a measurement of what the model tends to miss or not —
# see the error-analysis writeup for the actually-measured failure patterns.
SMALL_AREA_PX = 100.0
LARGE_AREA_PX = 1000.0

_MODEL_LIMITATIONS_NOTE = (
    "Note on reliability: this model is known (from held-out test-set error "
    "analysis) to sometimes miss very small changed objects, and can "
    "occasionally flag bare soil or construction-like surfaces as false "
    "changes. Treat flagged regions as candidates for review, not as a "
    "final verdict."
)


def _classify_size(area_px: float) -> str:
    if area_px < SMALL_AREA_PX:
        return "small"
    if area_px > LARGE_AREA_PX:
        return "large"
    return "medium"


def summarize_geojson(geojson: dict[str, Any]) -> dict[str, Any]:
    """Extract plain structured facts from a change-detection GeoJSON.

    Parameters
    ----------
    geojson:
        A FeatureCollection as produced by
        ``geoseg.inference.polygonize.polygons_to_geojson``.

    Returns
    -------
    A dict with:
      - ``n_regions``: number of change regions (features).
      - ``total_area_px``: sum of all region areas, in pixel^2.
      - ``largest_area_px`` / ``smallest_area_px``: extremes among regions
        (``0.0`` for both if there are no regions).
      - ``mean_area_px``: average region area (``0.0`` if there are no
        regions).
      - ``largest_region_size_class``: ``"small"``/``"medium"``/``"large"``
        classification of the largest region (``None`` if there are no
        regions) — see the module docstring for the caveat on these
        buckets being a readability heuristic, not a measured threshold.
      - ``image_name``: taken from the first feature's properties, if
        present, else ``None``.
    """
    features = geojson.get("features", [])
    areas = [float(f["properties"].get("area_px", 0.0)) for f in features]

    image_name = None
    if features:
        image_name = features[0].get("properties", {}).get("image_name")

    if not areas:
        return {
            "n_regions": 0,
            "total_area_px": 0.0,
            "largest_area_px": 0.0,
            "smallest_area_px": 0.0,
            "mean_area_px": 0.0,
            "largest_region_size_class": None,
            "image_name": image_name,
        }

    largest = max(areas)
    return {
        "n_regions": len(areas),
        "total_area_px": sum(areas),
        "largest_area_px": largest,
        "smallest_area_px": min(areas),
        "mean_area_px": sum(areas) / len(areas),
        "largest_region_size_class": _classify_size(largest),
        "image_name": image_name,
    }


def generate_report(
    geojson: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    image_name: str | None = None,
    include_caveat: bool = True,
) -> str:
    """Render a short natural-language report from a GeoJSON (and,
    optionally, test-set metrics).

    Parameters
    ----------
    geojson:
        A FeatureCollection as produced by
        ``geoseg.inference.polygonize.polygons_to_geojson``.
    metrics:
        Optional dict with test-set metrics, e.g. the contents of
        ``test_metrics.json`` (expects ``iou`` and/or ``f1`` keys if
        given). Used to add one sentence of overall model-reliability
        context; omitted entirely if not provided.
    image_name:
        Optional override for the image name shown in the report. If not
        given, falls back to the ``image_name`` found in the GeoJSON's
        feature properties (if any).
    include_caveat:
        Whether to append the fixed model-limitations note at the end.
        Defaults to True; set to False for a bare summary (e.g. if a
        caller wants to compose its own caveat wording).

    Returns
    -------
    A short multi-sentence report as plain text.
    """
    summary = summarize_geojson(geojson)
    name = image_name if image_name is not None else summary["image_name"]
    subject = f"In {name}, " if name else "In this image pair, "

    if summary["n_regions"] == 0:
        sentences = [f"{subject}no significant changes were detected."]
    else:
        n = summary["n_regions"]
        region_word = "region" if n == 1 else "regions"
        sentences = [
            f"{subject}{n} change {region_word} were detected, "
            f"covering a total area of about {summary['total_area_px']:.0f} px^2."
        ]
        size_class = summary["largest_region_size_class"]
        sentences.append(
            f"The largest region is about {summary['largest_area_px']:.0f} px^2 ({size_class})."
        )
        if n > 1:
            sentences.append(
                f"On average, each region covers about {summary['mean_area_px']:.0f} px^2, "
                f"ranging from {summary['smallest_area_px']:.0f} to "
                f"{summary['largest_area_px']:.0f} px^2."
            )

    if metrics:
        iou = metrics.get("iou")
        f1 = metrics.get("f1")
        if iou is not None and f1 is not None:
            sentences.append(
                f"For reference, on the held-out test set this model achieves an "
                f"IoU of {iou:.2f} and F1 of {f1:.2f}, so individual predictions "
                f"should be read with that overall accuracy in mind."
            )

    if include_caveat:
        sentences.append(_MODEL_LIMITATIONS_NOTE)

    return " ".join(sentences)
