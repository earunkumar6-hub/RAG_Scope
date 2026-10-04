"""GraphStore contract tests, run against every backend.

The Neo4j case runs only when ``NEO4J_TEST_URI`` is set (e.g. bolt://localhost:7687 for the
docker-compose service); it uses a throwaway label namespace and wipes it afterwards.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from app.graph.base import (
    EntityIn,
    GraphMismatchError,
    GraphStore,
    RelationIn,
    normalize_name,
    query_ngrams,
)
from app.graph.networkx_store import NetworkXGraphStore


def _neo4j_store() -> GraphStore:
    from app.graph.neo4j_store import Neo4jGraphStore

    return Neo4jGraphStore(
        os.environ["NEO4J_TEST_URI"],
        os.environ.get("NEO4J_TEST_USER", "neo4j"),
        os.environ.get("NEO4J_TEST_PASSWORD", "graphrag-test"),
        namespace="pytest",
    )


@pytest.fixture(params=["networkx", "neo4j"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[GraphStore]:
    if request.param == "networkx":
        yield NetworkXGraphStore(tmp_path)
        return
    if not os.environ.get("NEO4J_TEST_URI"):
        pytest.skip("NEO4J_TEST_URI not set")
    s = _neo4j_store()
    s.wipe()
    yield s
    s.wipe()
    s.close()


def E(name: str, chunk: str, type_: str = "CONCEPT", aliases: list[str] | None = None):  # noqa: N802
    return EntityIn(name, type_, f"about {name}", chunk, aliases or [])


def R(src: str, rel: str, tgt: str, chunk: str) -> RelationIn:  # noqa: N802
    return RelationIn(normalize_name(src), rel, normalize_name(tgt), chunk)


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_normalize_and_ngrams() -> None:
    assert normalize_name("The Knowledge-Graph!") == "knowledge graph"
    assert normalize_name("  ACME Corp. ") == "acme corp"
    grams = query_ngrams("What is the Knowledge Graph of ACME?")
    assert {"knowledge graph", "acme", "knowledge graph of acme"} <= grams


def test_upsert_merges_by_key_and_alias(store: GraphStore) -> None:
    r1 = store.upsert(
        [E("Knowledge Graph", "c1", aliases=["KG"]), E("Neo4j", "c1", "TECHNOLOGY")],
        [R("Knowledge Graph", "stored in", "Neo4j", "c1")],
    )
    assert (r1.added_vertices, r1.added_edges) == (2, 1)
    # same entity by a different surface form and by its alias, from another chunk
    r2 = store.upsert(
        [E("knowledge graph", "c2"), E("KG", "c3"), E("Neo4j", "c2", "TECHNOLOGY")],
        [R("KG", "stored in", "Neo4j", "c3"), R("Neo4j", "hosts", "KG", "c3")],
    )
    assert r2.added_vertices == 0
    assert r2.added_edges == 1  # "hosts" is new; "stored in" merged
    stats = store.stats()
    assert (stats["vertices"], stats["edges"]) == (2, 2)
    assert stats["types"] == {"CONCEPT": 1, "TECHNOLOGY": 1}
    sub = store.neighbourhood(["knowledge graph"], hops=1)
    kg = next(v for v in sub.vertices if v.id == "knowledge graph")
    assert kg.chunk_ids == ["c1", "c2", "c3"]
    assert "KG" in kg.aliases
    stored_in = next(e for e in sub.edges if e.relation == "stored in")
    assert stored_in.chunk_ids == ["c1", "c3"]


def test_relations_to_unknown_entities_are_dropped(store: GraphStore) -> None:
    res = store.upsert([E("A thing", "c1")], [R("A thing", "likes", "Ghost", "c1")])
    assert res.added_edges == 0
    assert store.stats()["edges"] == 0


def test_remove_chunks_strips_provenance(store: GraphStore) -> None:
    store.upsert(
        [E("Alpha", "c1"), E("Beta", "c1"), E("Alpha", "c2"), E("Gamma", "c2")],
        [R("Alpha", "links", "Beta", "c1"), R("Alpha", "links", "Gamma", "c2")],
    )
    removed = store.remove_chunks(["c1"])
    assert removed == (1, 1)  # Beta and its edge; Alpha survives via c2
    sub = store.neighbourhood(["alpha"], hops=2)
    assert sorted(v.id for v in sub.vertices) == ["alpha", "gamma"]
    assert next(v for v in sub.vertices if v.id == "alpha").chunk_ids == ["c2"]
    assert store.remove_chunks([]) == (0, 0)


def test_remove_chunks_moves_replaced_provenance(store: GraphStore) -> None:
    store.upsert(
        [E("Alpha", "c1"), E("Beta", "c1"), E("Alpha", "c9"), E("Gamma", "c3")],
        [R("Alpha", "links", "Beta", "c1"), R("Beta", "links", "Gamma", "c3")],
    )
    # c1 hands its facts to c9 (already on Alpha: no duplicate id); c3 is just dropped
    assert store.remove_chunks(["c1", "c3"], {"c1": "c9"}) == (1, 1)
    sub = store.neighbourhood(["alpha"], hops=2)
    by = {v.id: v.chunk_ids for v in sub.vertices}
    assert by == {"alpha": ["c9"], "beta": ["c9"]}
    assert [e.chunk_ids for e in sub.edges] == [["c9"]]


def test_match_exact_find_and_neighbourhood(store: GraphStore) -> None:
    store.upsert(
        [
            E("Knowledge Graph", "c1", aliases=["KG"]),
            E("Vertex", "c1"),
            E("Edge", "c1"),
            E("Traversal", "c2"),
        ],
        [
            R("Knowledge Graph", "has", "Vertex", "c1"),
            R("Vertex", "connected by", "Edge", "c1"),
            R("Traversal", "follows", "Edge", "c2"),
        ],
    )
    assert store.match_exact("How is a KG stored?") == ["knowledge graph"]
    assert store.match_exact("vertex and edge") == ["edge", "vertex"]
    assert store.find("graph") == ["knowledge graph"]
    one = store.neighbourhood(["knowledge graph"], hops=1)
    assert one.hops == {"knowledge graph": 0, "vertex": 1}
    two = store.neighbourhood(["knowledge graph"], hops=2)
    assert two.hops["edge"] == 2 and "traversal" not in two.hops
    assert len(store.neighbourhood(["knowledge graph"], hops=3, limit=2).vertices) == 2
    assert store.neighbourhood(["nope"], hops=2).vertices == []


def test_vectors_search_and_model_guard(store: GraphStore) -> None:
    store.upsert([E("Alpha", "c1"), E("Beta", "c1")], [])
    missing = dict(store.keys_without_vectors())
    assert missing == {"alpha": "Alpha", "beta": "Beta"}
    store.set_vectors("m1", ["alpha", "beta"], np.vstack([unit(1, 0), unit(0, 1)]))
    assert store.keys_without_vectors() == []
    hits = store.search_vectors("m1", unit(1, 0.1), k=5, min_similarity=0.5)
    assert [k for k, _ in hits] == ["alpha"]
    assert hits[0][1] == pytest.approx(0.995, abs=1e-3)
    assert store.search_vectors("other-model", unit(1, 0), k=5, min_similarity=0.0) == []
    with pytest.raises(GraphMismatchError):
        store.set_vectors("m2", ["alpha"], np.vstack([unit(1, 0)]))
    store.remove_chunks(["c1"])
    assert store.search_vectors("m1", unit(1, 0), k=5, min_similarity=0.0) == []


def test_overview_filters_types_and_orders_by_degree(store: GraphStore) -> None:
    store.upsert(
        [E("Hub", "c1"), E("Leaf One", "c1", "PERSON"), E("Leaf Two", "c1", "PERSON")],
        [R("Hub", "knows", "Leaf One", "c1"), R("Hub", "knows", "Leaf Two", "c1")],
    )
    ov = store.overview(None, limit=2)
    assert ov.vertices[0].id == "hub" and ov.vertices[0].degree == 2
    people = store.overview(["PERSON"], limit=10)
    assert sorted(v.id for v in people.vertices) == ["leaf one", "leaf two"]
    assert people.edges == []  # edges need both endpoints in the view


def test_networkx_persists_across_instances(tmp_path: Path) -> None:
    s = NetworkXGraphStore(tmp_path)
    s.upsert([E("Alpha", "c1", aliases=["A1"]), E("Beta", "c1")], [R("Alpha", "x", "Beta", "c1")])
    s.set_vectors("m1", ["alpha", "beta"], np.vstack([unit(1, 0), unit(0, 1)]))
    s.flush()
    t = NetworkXGraphStore(tmp_path)
    assert t.stats() == s.stats()
    assert t.match_exact("about a1") == ["alpha"]
    assert [k for k, _ in t.search_vectors("m1", unit(0, 1), 1, 0.5)] == ["beta"]
