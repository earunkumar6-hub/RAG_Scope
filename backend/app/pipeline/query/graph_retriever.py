"""Q5 graph retrieval: query -> matched vertices -> ``graph_hops`` neighbourhood -> evidence chunks.

Vertices are matched two ways: exact (a vertex name or alias appears as a phrase in the query)
and by embedding (query entities, extracted by the LLM, against entity-name vectors; without an
LLM the query vector itself is the probe). Evidence chunks are ranked by how many neighbourhood
vertices/edges cite them, weighted by 1 / (1 + hop distance).
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.graph.base import GraphStore, Subgraph
from app.llm.base import LLMError, LLMProvider
from app.llm.cache import cached_json
from app.pipeline.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

ENTITY_MATCH_MIN = 0.75
PER_PROBE = 3
MAX_SEEDS = 10
SUBGRAPH_LIMIT = 200
MAX_TRIPLES = 30

QUERY_ENTITY_SYSTEM = (
    "You list the entities mentioned in the question: named things, concepts, products, "
    'people, places. Reply with a JSON object {"entities": [str]} using the wording of the '
    "question, at most 5, and nothing else."
)


@dataclass
class GraphMatch:
    key: str
    method: str  # "exact" | "embedding"
    score: float
    probe: str


@dataclass
class GraphResult:
    query_entities: list[str]
    extraction_note: str
    matches: list[GraphMatch]
    subgraph: Subgraph
    evidence: list[tuple[str, float, list[str]]] = field(default_factory=list)
    triples: list[dict[str, Any]] = field(default_factory=list)


def extract_query_entities(
    llm_factory: Callable[[], LLMProvider], query: str, seed: int
) -> tuple[list[str], str]:
    """LLM-extracted entity strings, or ([], reason) when the LLM is unavailable or fails.

    Results are cached per (model, normalised query) so repeated queries match identically.
    """
    try:
        llm = llm_factory()
        data, hit = cached_json(
            "q5_query_entities",
            llm.fast_model,
            query,
            lambda: llm.complete_json(
                [
                    {"role": "system", "content": QUERY_ENTITY_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                seed=seed,
                max_tokens=100,
                fast=True,
            ),
        )
    except Exception as exc:  # optional step: fall back to the query vector
        msg = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
        return [], f"LLM entity extraction unavailable ({msg[:200]}); matched with the query vector"
    raw = data.get("entities", [])
    names = [" ".join(str(e).split()) for e in raw if isinstance(e, str) and e.strip()][:5]
    return names, "entities extracted by the LLM" + (" (cached)" if hit else "")


def match_vertices(
    store: GraphStore,
    embedder: Embedder,
    query: str,
    entities: list[str],
    query_vector: np.ndarray,
) -> list[GraphMatch]:
    best: dict[str, GraphMatch] = {
        k: GraphMatch(k, "exact", 1.0, query) for k in store.match_exact(query)
    }
    if entities:
        probes = list(zip(entities, embedder.embed_documents(entities), strict=True))
    else:
        probes = [(query, query_vector)]
    for probe, vector in probes:
        for key, sim in store.search_vectors(
            embedder.model_name, vector, PER_PROBE, ENTITY_MATCH_MIN
        ):
            current = best.get(key)
            if current is None or (current.method == "embedding" and sim > current.score):
                best[key] = GraphMatch(key, "embedding", sim, probe)
    ranked = sorted(best.values(), key=lambda m: (m.method != "exact", -m.score, m.key))
    return ranked[:MAX_SEEDS]


def rank_evidence(sub: Subgraph) -> list[tuple[str, float, list[str]]]:
    """(chunk_id, score, via entity names) best first."""
    score: dict[str, float] = defaultdict(float)
    via: dict[str, set[str]] = defaultdict(set)
    for v in sub.vertices:
        w = 1.0 / (1 + sub.hops.get(v.id, 0))
        for cid in v.chunk_ids:
            score[cid] += w
            via[cid].add(v.name)
    names = {v.id: v.name for v in sub.vertices}
    for e in sub.edges:
        w = 1.0 / (1 + max(sub.hops.get(e.source, 0), sub.hops.get(e.target, 0)))
        for cid in e.chunk_ids:
            score[cid] += w
            via[cid].update((names.get(e.source, e.source), names.get(e.target, e.target)))
    return sorted(
        ((cid, round(s, 4), sorted(via[cid])) for cid, s in score.items()),
        key=lambda t: (-t[1], t[0]),
    )


def triples(sub: Subgraph) -> list[dict[str, Any]]:
    names = {v.id: v.name for v in sub.vertices}
    rows = [
        {
            "source": names.get(e.source, e.source),
            "relation": e.relation,
            "target": names.get(e.target, e.target),
            "hop": max(sub.hops.get(e.source, 0), sub.hops.get(e.target, 0)),
            "chunk_count": len(e.chunk_ids),
        }
        for e in sub.edges
    ]
    rows.sort(key=lambda r: (r["hop"], r["source"], r["relation"], r["target"]))
    return rows[:MAX_TRIPLES]


def retrieve(
    store: GraphStore,
    embedder: Embedder,
    llm_factory: Callable[[], LLMProvider],
    query: str,
    query_vector: np.ndarray,
    hops: int,
    seed: int,
) -> GraphResult:
    entities, note = extract_query_entities(llm_factory, query, seed)
    matches = match_vertices(store, embedder, query, entities, query_vector)
    sub = store.neighbourhood([m.key for m in matches], hops, SUBGRAPH_LIMIT)
    return GraphResult(entities, note, matches, sub, rank_evidence(sub), triples(sub))
