"""Local ``GraphStore``: a NetworkX MultiDiGraph persisted to ``graph.json``, with entity-name
vectors in ``graph_vectors.npz``. Used when ``NEO4J_URI`` is not set. Writes are atomic
(temp file + rename).
"""

import json
import os
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from app.graph.base import (
    Edge,
    EntityIn,
    GraphMismatchError,
    GraphStore,
    RelationIn,
    Subgraph,
    UpsertResult,
    Vertex,
    normalize_name,
    query_ngrams,
)

FORMAT_VERSION = 1


def normalize_relation(relation: str) -> str:
    return " ".join(relation.strip().lower().split())[:60]


class NetworkXGraphStore(GraphStore):
    backend = "networkx"

    def __init__(self, data_dir: Path):
        self._path = data_dir / "graph.json"
        self._vec_path = data_dir / "graph_vectors.npz"
        self._lock = threading.RLock()
        self._g = nx.MultiDiGraph()
        self._alias: dict[str, str] = {}  # alias key -> vertex key
        self._vec_model: str | None = None
        self._vec_keys: list[str] = []
        self._vecs = np.zeros((0, 0), dtype=np.float32)
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        for n in data["nodes"]:
            self._g.add_node(
                n["id"],
                name=n["name"],
                type=n["type"],
                description=n["description"],
                aliases=set(n["aliases"]),
                chunk_ids=set(n["chunk_ids"]),
            )
        for e in data["edges"]:
            self._g.add_edge(
                e["source"], e["target"], key=e["relation"], chunk_ids=set(e["chunk_ids"])
            )
        self._alias = dict(data.get("aliases", {}))
        self._vec_model = data.get("vector_model")
        if self._vec_path.exists():
            with np.load(self._vec_path) as z:
                self._vec_keys = [str(k) for k in z["keys"]]
                self._vecs = z["vectors"].astype(np.float32)

    def flush(self) -> None:
        with self._lock:
            data = {
                "version": FORMAT_VERSION,
                "vector_model": self._vec_model,
                "nodes": [
                    {
                        "id": k,
                        "name": a["name"],
                        "type": a["type"],
                        "description": a["description"],
                        "aliases": sorted(a["aliases"]),
                        "chunk_ids": sorted(a["chunk_ids"]),
                    }
                    for k, a in self._g.nodes(data=True)
                ],
                "edges": [
                    {"source": s, "target": t, "relation": r, "chunk_ids": sorted(a["chunk_ids"])}
                    for s, t, r, a in self._g.edges(keys=True, data=True)
                ],
                "aliases": self._alias,
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self._path)
            vec_tmp = self._vec_path.with_name("graph_vectors.tmp.npz")
            np.savez(vec_tmp, keys=np.array(self._vec_keys, dtype=str), vectors=self._vecs)
            os.replace(vec_tmp, self._vec_path)

    # ------------------------------------------------------------------ writes
    def _resolve(self, key: str) -> str | None:
        if key in self._g:
            return key
        target = self._alias.get(key)
        return target if target in self._g else None

    def upsert(self, entities: list[EntityIn], relations: list[RelationIn]) -> UpsertResult:
        added_v = added_e = merged = 0
        with self._lock:
            for e in entities:
                key = e.key
                if not key:
                    continue
                target = self._resolve(key) or key
                if target in self._g:
                    attrs = self._g.nodes[target]
                    attrs["chunk_ids"].add(e.chunk_id)
                    if not attrs["description"] and e.description:
                        attrs["description"] = e.description
                    if attrs["type"] == "OTHER" and e.type != "OTHER":
                        attrs["type"] = e.type
                    if e.name != attrs["name"]:
                        attrs["aliases"].add(e.name)
                    merged += 1
                else:
                    self._g.add_node(
                        target,
                        name=e.name,
                        type=e.type,
                        description=e.description,
                        aliases=set(),
                        chunk_ids={e.chunk_id},
                    )
                    added_v += 1
                for alias in e.aliases:
                    ak = normalize_name(alias)
                    if ak and ak != target and ak not in self._g:
                        self._alias.setdefault(ak, target)
                        self._g.nodes[target]["aliases"].add(alias)
            for r in relations:
                src, tgt = self._resolve(r.source), self._resolve(r.target)
                rel = normalize_relation(r.relation)
                if src is None or tgt is None or src == tgt or not rel:
                    continue
                if self._g.has_edge(src, tgt, key=rel):
                    self._g.edges[src, tgt, rel]["chunk_ids"].add(r.chunk_id)
                else:
                    self._g.add_edge(src, tgt, key=rel, chunk_ids={r.chunk_id})
                    added_e += 1
        return UpsertResult(added_v, added_e, merged)

    def keys_without_vectors(self) -> list[tuple[str, str]]:
        with self._lock:
            have = set(self._vec_keys)
            return [(k, a["name"]) for k, a in self._g.nodes(data=True) if k not in have]

    def set_vectors(self, model: str, keys: list[str], vectors: np.ndarray) -> None:
        if not keys:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        with self._lock:
            if self._vec_model and self._vec_model != model:
                raise GraphMismatchError(
                    f"Graph entity vectors were built with {self._vec_model}, not {model}"
                )
            self._vec_model = model
            index = {k: i for i, k in enumerate(self._vec_keys)}
            new_keys = [k for k in keys if k not in index]
            position = {k: i for i, k in enumerate(keys)}
            rows = np.array([position[k] for k in new_keys], dtype=int)
            if len(new_keys):
                block = vectors[rows]
                self._vecs = block if self._vecs.size == 0 else np.vstack([self._vecs, block])
                self._vec_keys.extend(new_keys)

    def remove_chunks(
        self, chunk_ids: list[str], replace: dict[str, str] | None = None
    ) -> tuple[int, int]:
        replace = replace or {}
        gone = set(chunk_ids) | set(replace)
        removed_v = removed_e = 0
        if not gone:
            return 0, 0

        def strip(ids: set[str]) -> set[str]:
            return (ids - gone) | {replace[c] for c in ids & replace.keys()}

        with self._lock:
            for s, t, r, a in list(self._g.edges(keys=True, data=True)):
                a["chunk_ids"] = strip(a["chunk_ids"])
                if not a["chunk_ids"]:
                    self._g.remove_edge(s, t, key=r)
                    removed_e += 1
            dead = []
            for k, a in self._g.nodes(data=True):
                a["chunk_ids"] = strip(a["chunk_ids"])
                if not a["chunk_ids"]:
                    dead.append(k)
            for k in dead:
                removed_e += self._g.degree(k)
                self._g.remove_node(k)
            removed_v = len(dead)
            if dead:
                dead_set = set(dead)
                self._alias = {a: k for a, k in self._alias.items() if k not in dead_set}
                keep = [i for i, k in enumerate(self._vec_keys) if k not in dead_set]
                self._vec_keys = [self._vec_keys[i] for i in keep]
                self._vecs = self._vecs[keep] if self._vecs.size else self._vecs
        return removed_v, removed_e

    # ------------------------------------------------------------------ reads
    def _vertex(self, key: str) -> Vertex:
        a = self._g.nodes[key]
        return Vertex(
            id=key,
            name=a["name"],
            type=a["type"],
            description=a["description"],
            aliases=sorted(a["aliases"]),
            chunk_ids=sorted(a["chunk_ids"]),
            degree=self._g.degree(key),
        )

    def _subgraph(self, keys: list[str], hops: dict[str, int] | None = None) -> Subgraph:
        keep = set(keys)
        edges = [
            Edge(s, r, t, sorted(a["chunk_ids"]))
            for s, t, r, a in self._g.edges(keys=True, data=True)
            if s in keep and t in keep
        ]
        return Subgraph([self._vertex(k) for k in keys], edges, hops or {})

    def match_exact(self, text: str) -> list[str]:
        with self._lock:
            found = {self._resolve(g) for g in query_ngrams(text)} - {None}
            return sorted(k for k in found if len(k) > 1)

    def search_vectors(
        self, model: str, vector: np.ndarray, k: int, min_similarity: float
    ) -> list[tuple[str, float]]:
        with self._lock:
            if self._vecs.size == 0 or model != self._vec_model:
                return []
            sims = self._vecs @ np.asarray(vector, dtype=np.float32)
            order = np.argsort(-sims)[:k]
            return [(self._vec_keys[i], float(sims[i])) for i in order if sims[i] >= min_similarity]

    def neighbourhood(self, seeds: list[str], hops: int, limit: int = 300) -> Subgraph:
        with self._lock:
            dist: dict[str, int] = {}
            queue = deque()
            for s in seeds:
                if s in self._g and s not in dist:
                    dist[s] = 0
                    queue.append(s)
            undirected = self._g.to_undirected(as_view=True)
            while queue and len(dist) < limit:
                node = queue.popleft()
                if dist[node] >= hops:
                    continue
                for nb in sorted(undirected.neighbors(node)):
                    if nb not in dist and len(dist) < limit:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            return self._subgraph(list(dist), dist)

    def find(self, text: str, limit: int = 20) -> list[str]:
        needle = normalize_name(text)
        if not needle:
            return []
        with self._lock:
            hits = [
                k
                for k, a in self._g.nodes(data=True)
                if needle in k or any(needle in normalize_name(al) for al in a["aliases"])
            ]
            hits.sort(key=lambda k: (k != needle, -self._g.degree(k), k))
            return hits[:limit]

    def overview(self, types: list[str] | None, limit: int) -> Subgraph:
        with self._lock:
            keys = [k for k, a in self._g.nodes(data=True) if not types or a["type"] in types]
            keys.sort(key=lambda k: (-self._g.degree(k), k))
            return self._subgraph(keys[:limit])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            types = Counter(a["type"] for _, a in self._g.nodes(data=True))
            return {
                "vertices": self._g.number_of_nodes(),
                "edges": self._g.number_of_edges(),
                "types": dict(sorted(types.items())),
            }
