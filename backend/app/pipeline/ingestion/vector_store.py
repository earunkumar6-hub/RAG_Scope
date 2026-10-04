"""S7 Chroma vector store (PersistentClient, cosine space).

Chroma metadata cannot hold ``None``: ``is_duplicate_of=""`` means canonical, ``cluster_id=-1``
means unassigned. The collection records the embedding model and dimension it was built with;
ingesting with a different embedder fails fast instead of corrupting the index.
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

UPSERT_BATCH = 1000


class EmbeddingMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Neighbour:
    chunk_id: str
    similarity: float
    metadata: dict[str, Any]


def _where_not_in_documents(exclude_document_ids: list[str]) -> dict[str, Any] | None:
    return {"document_id": {"$nin": exclude_document_ids}} if exclude_document_ids else None


def _and(*clauses: dict[str, Any] | None) -> dict[str, Any] | None:
    present = [c for c in clauses if c]
    if not present:
        return None
    return present[0] if len(present) == 1 else {"$and": present}


class VectorStore:
    """Single shared Chroma client per data path (Chroma refuses a second client per path)."""

    def __init__(self, path: Path, collection_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._lock = threading.Lock()

    def heartbeat(self) -> int:
        return self._client.heartbeat()

    def _existing(self):  # noqa: ANN202
        try:
            return self._client.get_collection(self.collection_name, embedding_function=None)
        except Exception:  # chromadb raises NotFoundError (type differs across versions)
            return None

    def existing_collection(self):  # noqa: ANN201
        """The collection if it exists, else None (never creates it)."""
        return self._existing()

    def collection_info(self) -> dict[str, Any]:
        col = self._existing()
        if col is None:
            return {"name": self.collection_name, "exists": False, "count": 0}
        meta = col.metadata or {}
        return {
            "name": self.collection_name,
            "exists": True,
            "count": col.count(),
            "embedding_model": meta.get("embedding_model"),
            "embedding_dim": meta.get("embedding_dim"),
        }

    def collection(self, embedding_model: str, dimension: int):  # noqa: ANN201
        """Get or create the collection, verifying it was built with the same embedder."""
        with self._lock:
            col = self._existing()
            if col is None:
                return self._client.create_collection(
                    self.collection_name,
                    embedding_function=None,
                    metadata={
                        "hnsw:space": "cosine",
                        "embedding_model": embedding_model,
                        "embedding_dim": dimension,
                    },
                )
        return self._verified(col, embedding_model, dimension)

    def _verified(self, col, embedding_model: str, dimension: int):  # noqa: ANN001, ANN202
        meta = col.metadata or {}
        if meta.get("embedding_model") != embedding_model or meta.get("embedding_dim") != dimension:
            raise EmbeddingMismatchError(
                f"Collection '{self.collection_name}' was built with "
                f"{meta.get('embedding_model')} (dim {meta.get('embedding_dim')}), but the current "
                f"embedder is {embedding_model} (dim {dimension}). Reset the collection or switch "
                "the embedding provider back."
            )
        return col

    def search(
        self,
        vector: np.ndarray,
        n_results: int,
        embedding_model: str,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[Neighbour, str]]:
        """Top ``n_results`` (neighbour, text) for a query vector; [] if the collection is empty."""
        col = self._existing()
        if col is None or col.count() == 0:
            return []
        col = self._verified(col, embedding_model, len(vector))
        res = col.query(
            query_embeddings=[np.asarray(vector).tolist()],
            n_results=min(n_results, col.count()),
            where=where,
            include=["distances", "metadatas", "documents"],
        )
        return [
            (Neighbour(cid, 1.0 - float(dist), dict(meta or {})), text or "")
            for cid, dist, meta, text in zip(
                res["ids"][0],
                res["distances"][0],
                res["metadatas"][0],
                res["documents"][0],
                strict=True,
            )
        ]

    def cluster_centroids(self) -> np.ndarray:
        """L2-normalised mean vector of each topic cluster's canonical chunks, shape (k, dim)."""
        col = self._existing()
        if col is None or col.count() == 0:
            return np.zeros((0, 0), dtype=np.float32)
        res = col.get(where={"is_duplicate_of": ""}, include=["embeddings", "metadatas"])
        groups: dict[int, list[np.ndarray]] = {}
        for vec, meta in zip(res["embeddings"], res["metadatas"], strict=True):
            groups.setdefault(int((meta or {}).get("cluster_id", -1)), []).append(np.asarray(vec))
        centroids = np.vstack([np.mean(vs, axis=0) for _, vs in sorted(groups.items())])
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        return (centroids / np.clip(norms, 1e-12, None)).astype(np.float32)

    def get_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        """``{chunk_id: embedding}`` for the ids that exist."""
        col = self._existing()
        if col is None or not ids:
            return {}
        res = col.get(ids=ids, include=["embeddings"])
        return {cid: np.asarray(v) for cid, v in zip(res["ids"], res["embeddings"], strict=True)}

    def get_records(self, ids: list[str]) -> dict[str, tuple[dict[str, Any], str]]:
        """``{chunk_id: (metadata, text)}`` for the ids that exist."""
        col = self._existing()
        if col is None or not ids:
            return {}
        res = col.get(ids=ids, include=["metadatas", "documents"])
        return {
            cid: (dict(meta or {}), text or "")
            for cid, meta, text in zip(res["ids"], res["metadatas"], res["documents"], strict=True)
        }

    def count(self) -> int:
        col = self._existing()
        return col.count() if col is not None else 0

    def nearest(
        self,
        col,
        vectors: np.ndarray,
        exclude_document_ids: list[str],  # noqa: ANN001
    ) -> list[Neighbour | None]:
        """Nearest stored chunk for each vector (``None`` when the collection is empty)."""
        if len(vectors) == 0 or col.count() == 0:
            return [None] * len(vectors)
        res = col.query(
            query_embeddings=np.asarray(vectors).tolist(),
            n_results=1,
            where=_where_not_in_documents(exclude_document_ids),
            include=["distances", "metadatas"],
        )
        out: list[Neighbour | None] = []
        for ids, dists, metas in zip(res["ids"], res["distances"], res["metadatas"], strict=True):
            out.append(
                Neighbour(ids[0], 1.0 - float(dists[0]), dict(metas[0] or {})) if ids else None
            )
        return out

    def unique_records(
        self,
        col,
        exclude_document_ids: list[str],  # noqa: ANN001
    ) -> tuple[list[str], np.ndarray, list[str]]:
        """All canonical (non-duplicate) chunks: ids, vectors, texts."""
        if col.count() == 0:
            return [], np.zeros((0, 0)), []
        res = col.get(
            where=_and({"is_duplicate_of": ""}, _where_not_in_documents(exclude_document_ids)),
            include=["embeddings", "documents"],
        )
        embeddings = res["embeddings"]
        vectors = (
            np.asarray(embeddings)
            if embeddings is not None and len(embeddings)
            else (np.zeros((0, 0)))
        )
        return list(res["ids"]), vectors, list(res["documents"] or [])

    def upsert(  # noqa: ANN001
        self, col, ids: list[str], vectors: np.ndarray, texts: list[str], metadatas: list[dict]
    ) -> None:
        for i in range(0, len(ids), UPSERT_BATCH):
            sl = slice(i, i + UPSERT_BATCH)
            col.upsert(
                ids=ids[sl],
                embeddings=np.asarray(vectors[sl]).tolist(),
                documents=texts[sl],
                metadatas=metadatas[sl],
            )

    def update_metadata(self, col, ids: list[str], metadatas: list[dict]) -> None:  # noqa: ANN001
        """Merge metadata keys into existing records."""
        for i in range(0, len(ids), UPSERT_BATCH):
            col.update(ids=ids[i : i + UPSERT_BATCH], metadatas=metadatas[i : i + UPSERT_BATCH])

    def ids_for_documents(self, col, document_ids: list[str]) -> list[str]:  # noqa: ANN001
        if not document_ids:
            return []
        return list(col.get(where={"document_id": {"$in": document_ids}}, include=[])["ids"])

    def delete_ids(self, col, ids: list[str]) -> None:  # noqa: ANN001
        for i in range(0, len(ids), UPSERT_BATCH):
            col.delete(ids=ids[i : i + UPSERT_BATCH])
