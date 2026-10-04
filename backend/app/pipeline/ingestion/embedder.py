"""S5 embeddings: local sentence-transformers (default) or OpenAI, always L2-normalised."""

import os
import threading
from collections.abc import Callable
from typing import Protocol

import numpy as np

from app.core.config import Settings

BATCH_SIZE = 32
# bge-*-v1.5 retrieval instruction, applied to queries only (passages are embedded as-is).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
OPENAI_MAX_INPUT_TOKENS = 8191

ProgressFn = Callable[[int, int], None]  # (done, total)


class Embedder(Protocol):
    model_name: str

    @property
    def loaded(self) -> bool: ...

    @property
    def dimension(self) -> int: ...

    def load(self) -> None: ...

    def embed_documents(
        self, texts: list[str], on_batch: ProgressFn | None = None
    ) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def count_truncated(self, texts: list[str]) -> tuple[int, int]:
        """(number of texts longer than the model's input limit, that limit in model tokens)."""
        ...


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def _batched(
    texts: list[str], embed: Callable[[list[str]], np.ndarray], on_batch: ProgressFn | None
) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        out.append(embed(texts[i : i + BATCH_SIZE]))
        if on_batch:
            on_batch(min(i + BATCH_SIZE, len(texts)), len(texts))
    return l2_normalise(np.vstack(out)) if out else np.zeros((0, 0), dtype=np.float32)


class SentenceTransformerEmbedder:
    """Local CPU embedder; the model loads lazily (first call) and is shared across requests."""

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
                # Plain-text progress bars would break the JSON-only log stream.
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self._device)

    @property
    def _st(self):  # noqa: ANN202
        self.load()
        return self._model

    @property
    def dimension(self) -> int:
        model = self._st
        getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        return int(getter())

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._st.encode(
            texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False
        )

    def embed_documents(self, texts: list[str], on_batch: ProgressFn | None = None) -> np.ndarray:
        return _batched(texts, self._encode, on_batch)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        return l2_normalise(self._encode([prefix + text]))[0]

    def count_truncated(self, texts: list[str]) -> tuple[int, int]:
        model = self._st
        limit = int(model.max_seq_length)
        lengths = [len(ids) for ids in model.tokenizer(texts, add_special_tokens=True)["input_ids"]]
        return sum(1 for n in lengths if n > limit), limit


class OpenAIEmbedder:
    """OpenAI embeddings API (``EMBEDDING_PROVIDER=openai``)."""

    def __init__(self, settings: Settings):
        self.model_name = settings.openai_embedding_model
        self._settings = settings
        self._client = None
        self._dimension: int | None = None

    @property
    def _api(self):  # noqa: ANN202
        if self._client is None:
            from openai import OpenAI

            if self._settings.openai_api_key is None:
                raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
            self._client = OpenAI(
                api_key=self._settings.openai_api_key.get_secret_value(),
                base_url=self._settings.openai_base_url,
            )
        return self._client

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.embed_query("dimension probe").shape[0])
        return self._dimension

    def _encode(self, texts: list[str]) -> np.ndarray:
        resp = self._api.embeddings.create(model=self.model_name, input=texts)
        return np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])

    def embed_documents(self, texts: list[str], on_batch: ProgressFn | None = None) -> np.ndarray:
        return _batched(texts, self._encode, on_batch)

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(self._encode([text]))[0]

    def count_truncated(self, texts: list[str]) -> tuple[int, int]:
        from app.pipeline.ingestion.tokenizer import get_tokenizer

        tok = get_tokenizer("cl100k_base")
        return (
            sum(1 for t in texts if tok.count(t) > OPENAI_MAX_INPUT_TOKENS),
            OPENAI_MAX_INPUT_TOKENS,
        )


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    return SentenceTransformerEmbedder(settings.embedding_model)


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Deterministic 2-D PCA projection (sign-fixed so plots do not flip between runs)."""
    n = vectors.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    for i in range(components.shape[0]):
        if components[i][np.argmax(np.abs(components[i]))] < 0:
            components[i] = -components[i]
    projected = centered @ components.T
    if projected.shape[1] == 1:
        projected = np.hstack([projected, np.zeros((n, 1))])
    return projected
