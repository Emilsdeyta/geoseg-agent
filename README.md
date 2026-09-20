# GeoSeg-Agent

Satellite image segmentation, change detection and agent-based geospatial analysis.

> Status: work in progress

## Quick start

```bash
python -m venv .venv
.venv\\Scripts\\activate      # Linux/Mac: source .venv/bin/activate
pip install -e ".[dev,api]"
pytest
```

## Results
_TBD: mIoU / F1 table, before-after visuals, error analysis_

## Roadmap
- [ ] Tile-based data pipeline
- [ ] Baseline U-Net
- [ ] SegFormer + ablations
- [ ] Change detection
- [ ] Agent with tool calling
- [ ] FastAPI + Docker (GPU)
