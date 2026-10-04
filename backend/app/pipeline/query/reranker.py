"""Q7 cross-encoder re-ranker (local sentence-transformers ``CrossEncoder``, loaded lazily)."""

import os
import threading
from typing import Protocol

BATCH_SIZE = 16
# Each (query, passage) pair is truncated to this many model tokens. Cross-encoder cost grows
# with sequence length; 256 roughly halves CPU time versus the model max (512) for long chunks.
MAX_PAIR_TOKENS = 256


class Reranker(Protocol):
    model_name: str

    @property
    def loaded(self) -> bool: ...

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score per text (higher is more relevant)."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name, device=self._device, max_length=MAX_PAIR_TOKENS
                )

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        self.load()
        scores = self._model.predict(  # sigmoid-activated for single-label models: 0..1
            [(query, t) for t in texts], batch_size=BATCH_SIZE, show_progress_bar=False
        )
        return [float(s) for s in scores]
