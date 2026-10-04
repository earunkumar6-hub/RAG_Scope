"""``GraphStore`` interface for the knowledge graph (S8 build, Q5 retrieval, ``GET /api/graph``).

Vertices are entities keyed by their normalised name; edges are relations keyed by
(source, relation, target). Every vertex and edge carries the chunk ids it was extracted from
(provenance), so evidence chunks can be traced and re-chunked documents cleanly removed.
Implementations: NetworkX (local JSON file) and Neo4j. Others (e.g. Spanner Graph) can plug in.
"""

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "PRODUCT",
    "TECHNOLOGY",
    "CONCEPT",
    "EVENT",
    "OTHER",
)
MAX_NGRAM = 6
_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+")


class GraphMismatchError(RuntimeError):
    """The stored entity vectors were built with a different embedding model."""


def normalize_name(name: str) -> str:
    """Case-folded, accent-stable key: punctuation to spaces, leading article dropped."""
    text = unicodedata.normalize("NFKC", name).casefold()
    text = re.sub(r"[^\w]+", " ", text).strip()
    return _LEADING_ARTICLE.sub("", text)


def query_ngrams(text: str, max_n: int = MAX_NGRAM) -> set[str]:
    """All normalised word n-grams (n <= ``max_n``) of ``text``, for exact entity matching."""
    words = normalize_name(text).split()
    grams = set()
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            grams.add(_LEADING_ARTICLE.sub("", " ".join(words[i : i + n])))
    grams.discard("")
    return grams


@dataclass
class EntityIn:
    """An entity as extracted from one chunk."""

    name: str
    type: str
    description: str
    chunk_id: str
    aliases: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return normalize_name(self.name)


@dataclass
class RelationIn:
    source: str  # entity key
    relation: str
    target: str  # entity key
    chunk_id: str


@dataclass
class Vertex:
    id: str  # normalised key
    name: str
    type: str
    description: str
    aliases: list[str]
    chunk_ids: list[str]
    degree: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": self.aliases,
            "chunk_ids": self.chunk_ids,
            "degree": self.degree,
        }

    def summary(self) -> dict[str, Any]:
        """Compact form for stage events: provenance as a count (lists can be huge)."""
        d = self.as_dict()
        d["chunk_count"] = len(d.pop("chunk_ids"))
        return d


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    chunk_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "chunk_ids": self.chunk_ids,
        }

    def summary(self) -> dict[str, Any]:
        d = self.as_dict()
        d["chunk_count"] = len(d.pop("chunk_ids"))
        return d


@dataclass
class Subgraph:
    vertices: list[Vertex]
    edges: list[Edge]
    hops: dict[str, int] = field(default_factory=dict)  # vertex id -> distance from a seed


@dataclass
class UpsertResult:
    added_vertices: int
    added_edges: int
    merged_vertices: int


class GraphStore(ABC):
    backend: str

    @abstractmethod
    def upsert(self, entities: list[EntityIn], relations: list[RelationIn]) -> UpsertResult:
        """Merge entities/relations; aliases map onto existing vertices; provenance is unioned."""

    @abstractmethod
    def keys_without_vectors(self) -> list[tuple[str, str]]:
        """(key, name) of vertices whose name vector has not been stored yet."""

    @abstractmethod
    def set_vectors(self, model: str, keys: list[str], vectors: np.ndarray) -> None:
        """Store L2-normalised name vectors; raises ``GraphMismatchError`` on a model change."""

    @abstractmethod
    def remove_chunks(
        self, chunk_ids: list[str], replace: dict[str, str] | None = None
    ) -> tuple[int, int]:
        """Strip provenance; delete vertices/edges left without any. Returns (vertices, edges)
        removed. Ids in ``replace`` are swapped for their replacement instead of dropped (a
        deleted canonical chunk handing its facts to the duplicate promoted in its place)."""

    @abstractmethod
    def match_exact(self, text: str) -> list[str]:
        """Keys of vertices whose name or an alias appears as a phrase in ``text``."""

    @abstractmethod
    def search_vectors(
        self, model: str, vector: np.ndarray, k: int, min_similarity: float
    ) -> list[tuple[str, float]]:
        """Nearest vertices by name embedding: (key, cosine similarity), best first."""

    @abstractmethod
    def neighbourhood(self, seeds: list[str], hops: int, limit: int = 300) -> Subgraph:
        """Vertices within ``hops`` of ``seeds`` (BFS, either edge direction) and edges among
        them; at most ``limit`` vertices, nearest first."""

    @abstractmethod
    def find(self, text: str, limit: int = 20) -> list[str]:
        """Keys whose name or alias contains ``text`` (normalised substring), for search."""

    @abstractmethod
    def overview(self, types: list[str] | None, limit: int) -> Subgraph:
        """Highest-degree vertices (optionally of ``types``) and the edges among them."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """{"vertices": int, "edges": int, "types": {type: count}}"""

    def flush(self) -> None:  # noqa: B027 - optional hook
        """Persist pending writes (no-op for stores that write through)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (connections)."""
