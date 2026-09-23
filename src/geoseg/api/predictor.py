"""Model wrapper used by the API.

Kept separate from ``geoseg.api.main`` so the web layer can be imported and
tested without torch: the API only depends on the small :class:`ChangePredictor`
protocol, and :class:`TorchPredictor` imports torch lazily.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ChangePredictor(Protocol):
    """Anything that can turn a before/after RGB pair into a binary change mask."""

    checkpoint_epoch: int | None
    default_threshold: float

    def predict_mask(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return a (H, W) boolean/uint8 mask of predicted change."""
        ...


class TorchPredictor:
    """Loads a training checkpoint once and serves tiled full-image inference."""

    def __init__(self, checkpoint_path: Path, *, overlap: int = 0) -> None:
        import torch

        from geoseg.evaluation.evaluate import load_checkpoint_config
        from geoseg.training.trainer import build_model

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        cfg = load_checkpoint_config(checkpoint, None)
        # The checkpoint already holds the trained weights. Skip the ImageNet
        # download so the server also starts offline (e.g. inside Docker).
        cfg.model.encoder_weights = None
        model = build_model(cfg).to(self._device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self._model: Any = model
        self._tile_size: int = cfg.data.tile_size
        self._overlap = overlap
        self._lock = threading.Lock()  # one forward pass at a time
        self.default_threshold: float = cfg.train.threshold
        self.checkpoint_epoch: int | None = checkpoint.get("epoch")
        logger.info(
            "Loaded checkpoint %s (epoch=%s, device=%s)",
            checkpoint_path,
            self.checkpoint_epoch,
            self._device,
        )

    def predict_mask(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        from geoseg.inference.predict import predict_full_image

        used = self.default_threshold if threshold is None else threshold
        with self._lock, self._torch.no_grad():
            mask: np.ndarray = predict_full_image(
                self._model,
                image_a,
                image_b,
                tile_size=self._tile_size,
                overlap=self._overlap,
                device=self._device,
                threshold=used,
            )
        return mask
