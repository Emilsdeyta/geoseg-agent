# CPU-only image: no GPU / paid cloud needed. Checkpoint is mounted, not baked in.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU wheels first so the later extras install does not pull the CUDA build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,geo,ml]"

ENV GEOSEG_CHECKPOINT=/models/best.pt \
    GEOSEG_METRICS=/models/test_metrics.json

EXPOSE 8000
CMD ["uvicorn", "geoseg.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
