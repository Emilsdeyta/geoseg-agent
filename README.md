# GeoSeg-Agent

[![CI](https://github.com/Emilsdeyta/geoseg-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Emilsdeyta/geoseg-agent/actions/workflows/ci.yml)

Satellite image segmentation, change detection and agent-based geospatial analysis.

> Status: work in progress

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate      # Linux/Mac: source .venv/bin/activate
pip install -e ".[dev,api]"
pytest
```

## Results
_TBD: mIoU / F1 table, before-after visuals, error analysis_

## Evaluation & Error Analysis

### Test set results (LEVIR-CD, 128 held-out image pairs, unseen during training)

| Metric | Tile-level (pooled) | Per-image (macro-avg) |
|---|---|---|
| IoU | **0.818** | 0.738 |
| F1 | **0.900** | — |
| Precision | 0.912 | — |
| Recall | 0.889 | — |

The two IoU numbers differ because they weight pixels differently, not because of a bug:
- **Pooled/micro** sums TP/FP/FN across all 2,048 tiles before computing one ratio — images
  with larger changed areas contribute more.
- **Per-image/macro** computes IoU per image and averages 128 equally-weighted scores.
  Images with very small or zero change area are disproportionately punished by a single
  wrong pixel, which pulls the macro-average down. This is expected and common on
  imbalanced change-detection datasets, not a sign of a worse model.

Validation (model-selection) score at the best checkpoint (epoch 42/50): IoU 0.827, F1 0.905.
Test is slightly lower (0.818 / 0.900), a normal, healthy generalization gap.

### Qualitative error analysis (best/worst 5 test images, ranked by per-image IoU)

Out of 128 test images, 9 scored exactly 0.0 IoU. Inspecting these and the best-scoring
images directly revealed two recurring failure patterns:

1. **Missed small/subtle change objects (false negatives).** In several of the worst cases
   (e.g. `test_128`, `test_59`, `test_61`) the ground-truth change region is only a handful
   of pixels — a small building or structure — and the model predicts no change at all.
   This is the dominant cause of the zero-IoU cases: very small objects are easy to miss
   given the ~5% overall change-pixel ratio in the dataset.
2. **False positives on bare soil / construction-like surfaces.** In `test_62`, a
   already-built residential block contains a few patches of light-colored bare ground
   (likely a construction or landscaping area), which the model flags as multiple sizeable
   false-positive change regions even though ground truth marks no change there.
3. **Occasional seasonal-appearance confusion (weaker pattern).** In `test_60`, a strong
   before/after color shift in farmland (bare soil → green vegetation) triggers one small
   false positive. Notably, a very similar seasonal shift in `test_64` (one of the
   *best*-scoring images) is correctly ignored by the model — so seasonal color change
   alone isn't a reliable failure predictor; it appears to interact with local texture/shape.

**Caveat on the "best" examples:** most of the top-scoring images (IoU = 1.0) are simply
easy true negatives — no real change present in either image, and the model correctly
predicts nothing. This metric convention (0/0 defined as a perfect score) means the "best"
list reflects trivial correct rejections rather than the model's hardest successes; it is
included for completeness, not as evidence of exceptional performance on hard cases.

### Takeaways for future work
- Small/tiny changed objects are the largest single error source — worth exploring
  tile-size/overlap tuning, a small-object-aware loss term, or oversampling tiles with
  small positive regions.
- Bare-soil/construction misclassification suggests the model may be relying partly on
  simple brightness/texture cues rather than more robust structural change signals —
  a candidate direction for hard-negative mining if pursued further (see optional
  ablation, Phase C).

*Full ranking (`ranking.json`) and the 10 exported comparison images (best/worst 5) are
available in `outputs/full_run/error_analysis/`.*

## Roadmap
- [ ] Tile-based data pipeline
- [ ] Baseline U-Net
- [ ] SegFormer + ablations
- [ ] Change detection
- [ ] Agent with tool calling
- [ ] FastAPI + Docker (GPU)
