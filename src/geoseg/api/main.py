"""FastAPI entrypoint."""

from fastapi import FastAPI

from geoseg import __version__

app = FastAPI(title="GeoSeg-Agent", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
