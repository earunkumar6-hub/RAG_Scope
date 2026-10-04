from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.graph.base import Edge, EntityIn, RelationIn, Subgraph, Vertex
from app.graph.networkx_store import NetworkXGraphStore
from app.llm.base import LLMNotConfigured
from app.pipeline.query.fusion import RRF_K, weighted_rrf
from app.pipeline.query.graph_retriever import (
    extract_query_entities,
    match_vertices,
    rank_evidence,
    triples,
)
from tests.helpers import FakeEmbedder, FakeLLM, FakeReranker, make_text
from tests.test_ingest_api import ingest_ok
from tests.test_query import run_query

QUERY = "How does cosine similarity relate to an embedding vector?"


# ---------------------------------------------------------------- fusion
def test_weighted_rrf_scores_and_sources() -> None:
    rows = weighted_rrf(["a", "b", "c"], ["c", "d"], weight_vector=0.7)
    by = {r.chunk_id: r for r in rows}
    assert by["c"].score == pytest.approx(0.7 / (RRF_K + 3) + 0.3 / (RRF_K + 1))
    assert by["d"].score == pytest.approx(0.3 / (RRF_K + 2))
    # c (in both lists) beats a: 0.7/63 + 0.3/61 = 0.0160 > 0.7/61 = 0.0115
    assert [r.chunk_id for r in rows] == ["c", "a", "b", "d"]
    assert [r.source for r in rows] == ["both", "vector", "vector", "graph"]


def test_weighted_rrf_extremes_and_dedup() -> None:
    assert [r.chunk_id for r in weighted_rrf(["a"], ["b"], 1.0)] == ["a"]  # vector only
    assert [r.chunk_id for r in weighted_rrf(["a"], ["b"], 0.0)] == ["b"]  # graph only
    rows = weighted_rrf(["a", "a", "b"], ["b", "b"], 0.5)
    assert [(r.chunk_id, r.vector_rank, r.graph_rank) for r in rows] == [
        ("b", 2, 1),
        ("a", 1, None),
    ]


# ---------------------------------------------------------------- graph retrieval units
def _sub() -> Subgraph:
    v = lambda i, c: Vertex(i, i.title(), "CONCEPT", "", [], c)  # noqa: E731
    return Subgraph(
        vertices=[v("seed", ["c1", "c2"]), v("near", ["c2"]), v("far", ["c3"])],
        edges=[Edge("seed", "links", "near", ["c2"]), Edge("near", "links", "far", ["c3"])],
        hops={"seed": 0, "near": 1, "far": 2},
    )


def test_rank_evidence_weights_by_hop() -> None:
    ranked = rank_evidence(_sub())
    # c2: seed (1) + near (1/2) + seed-near edge (1/2) = 2.0; c1: 1.0; c3: far 1/3 + edge 1/3
    assert [(cid, score) for cid, score, _ in ranked] == [("c2", 2.0), ("c1", 1.0), ("c3", 0.6667)]
    assert ranked[0][2] == ["Near", "Seed"]


def test_triples_sorted_by_hop() -> None:
    rows = triples(_sub())
    assert [(r["source"], r["target"], r["hop"]) for r in rows] == [
        ("Seed", "Near", 1),
        ("Near", "Far", 2),
    ]


def test_match_vertices_exact_and_embedding(tmp_path: Path) -> None:
    store, emb = NetworkXGraphStore(tmp_path), FakeEmbedder()
    store.upsert(
        [
            EntityIn("Cosine", "CONCEPT", "", "c1"),
            EntityIn("Embedding Vector", "CONCEPT", "", "c1"),
        ],
        [RelationIn("cosine", "compares", "embedding vector", "c1")],
    )
    keys = [k for k, _ in store.keys_without_vectors()]
    store.set_vectors(
        emb.model_name, keys, emb.embed_documents([n for _, n in store.keys_without_vectors()])
    )
    matches = match_vertices(
        store, emb, "what is cosine?", ["embedding vector"], emb.embed_query("x")
    )
    assert [(m.key, m.method) for m in matches] == [
        ("cosine", "exact"),
        ("embedding vector", "embedding"),
    ]
    assert matches[1].score == pytest.approx(1.0) and matches[1].probe == "embedding vector"


def test_extract_query_entities_falls_back_without_llm(app: FastAPI) -> None:  # app: DB for cache
    def no_llm():
        raise LLMNotConfigured("OPENAI_API_KEY not set")

    assert extract_query_entities(no_llm, QUERY, 42)[0] == []
    assert "query vector" in extract_query_entities(no_llm, QUERY, 42)[1]
    llm = FakeLLM()
    names, note = extract_query_entities(lambda: llm, QUERY, 42)
    assert names == ["Cosine", "Similarity", "Embedding", "Vector"] and "LLM" in note
    assert "cached" not in note
    # identical (modulo case/spacing) query: served from the cache, no second LLM call
    again, note2 = extract_query_entities(lambda: llm, "  " + QUERY.upper(), 42)
    assert again == names and note2.endswith("(cached)")
    assert len(llm.complete_calls) == 1


# ---------------------------------------------------------------- Q5/Q6 through the API
@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def graph_client(app: FastAPI, llm: FakeLLM) -> Iterator[TestClient]:
    app.state.embedder = FakeEmbedder()
    app.state.reranker = FakeReranker()
    app.state.llm_factory = lambda: llm
    with TestClient(app) as c:
        ingest_ok(c, [("notes.txt", make_text(seed=5, paragraphs=30).encode())])
        yield c


def test_q5_q6_end_to_end(graph_client: TestClient, llm: FakeLLM) -> None:
    stages, _, done = run_query(graph_client, {"query": QUERY, "params": {"graph_hops": 1}})
    q5, q6 = stages["Q5_graph_retrieve"], stages["Q6_fusion"]
    assert q5["status"] == "success" and q5["data"].get("skipped") is None, q5["summary"]
    assert q5["params_used"] == {"graph_hops": 1, "graph_backend": "networkx"}
    methods = {m["id"]: m["method"] for m in q5["data"]["matches"]}
    assert methods["cosine"] == "exact" and methods["embedding"] == "exact"
    sub = q5["data"]["subgraph"]
    assert {v["id"] for v in sub["vertices"] if v["matched"]} == set(methods)
    assert max(v["hop"] for v in sub["vertices"]) <= 1
    assert q5["data"]["evidence"] and q5["data"]["triples"]

    rows = q6["data"]["rows"]
    assert q6["status"] == "success"
    assert {r["source"] for r in rows} <= {"vector", "graph", "both"}
    assert any(r["source"] in ("graph", "both") for r in rows)
    assert [r["fused_rank"] for r in rows] == list(range(1, len(rows) + 1))
    scores = [r["fused_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    # Q7 re-ranks exactly the top_k fused rows
    assert {r["chunk_id"] for r in stages["Q7_rerank"]["data"]["ranking"]} == {
        r["chunk_id"] for r in rows if r["kept"]
    }
    prompt = stages["Q8_generate"]["data"]["prompt"][1]["content"]
    assert prompt.startswith("Knowledge graph facts")
    assert stages["Q8_generate"]["data"]["facts_used"] == len(q5["data"]["triples"])
    assert done["status"] == "success"


def test_graph_hops_zero_is_vector_only(graph_client: TestClient) -> None:
    stages, _, _ = run_query(graph_client, {"query": QUERY, "params": {"graph_hops": 0}})
    assert stages["Q5_graph_retrieve"]["data"] == {"skipped": True}
    assert "disabled" in stages["Q5_graph_retrieve"]["summary"]
    assert stages["Q6_fusion"]["data"] == {"skipped": True}
    assert "Knowledge graph facts" not in stages["Q8_generate"]["data"]["prompt"][1]["content"]


def test_graph_candidates_bypass_similarity_threshold(
    graph_client: TestClient, llm: FakeLLM
) -> None:
    stages, _, done = run_query(
        graph_client, {"query": QUERY, "params": {"similarity_threshold": 1.0}}
    )
    assert stages["Q4_vector_retrieve"]["data"]["kept"] == 0
    rows = stages["Q6_fusion"]["data"]["rows"]
    assert rows and all(r["source"] == "graph" for r in rows)
    assert stages["Q8_generate"]["data"]["llm_called"] is True
    assert llm.calls  # the answer came from graph evidence


def test_vector_weight_one_drops_graph_only_rows(graph_client: TestClient) -> None:
    stages, _, _ = run_query(
        graph_client,
        {"query": QUERY, "params": {"hybrid_weight_vector": 1.0, "similarity_threshold": 0}},
    )
    assert {r["source"] for r in stages["Q6_fusion"]["data"]["rows"]} <= {"vector", "both"}


def test_document_filter_applies_to_graph_candidates(graph_client: TestClient) -> None:
    stages, _, _ = run_query(
        graph_client,
        {
            "query": QUERY,
            "filters": {"document_ids": ["nope"]},
            "params": {"similarity_threshold": 0},
        },
    )
    q6 = stages["Q6_fusion"]["data"]
    assert q6.get("rows", []) == []
    assert stages["Q8_generate"]["data"]["llm_called"] is False


# ---------------------------------------------------------------- GET /api/graph
def test_graph_api_overview_search_and_filters(graph_client: TestClient) -> None:
    full = graph_client.get("/api/graph").json()
    assert full["seeds"] == [] and full["vertices"] and full["edges"]
    assert full["stats"]["vertices"] >= len(full["vertices"])
    degrees = [v["degree"] for v in full["vertices"]]
    assert degrees == sorted(degrees, reverse=True)
    ids = {v["id"] for v in full["vertices"]}
    assert all(e["source"] in ids and e["target"] in ids for e in full["edges"])

    near = graph_client.get("/api/graph", params={"entity": "cosine", "hops": 1}).json()
    assert near["seeds"] == ["cosine"]
    hops = {v["id"]: v["hop"] for v in near["vertices"]}
    assert hops["cosine"] == 0 and set(hops.values()) <= {0, 1}

    seed_only = graph_client.get("/api/graph", params={"entity": "cosine", "hops": 0}).json()
    assert [v["id"] for v in seed_only["vertices"]] == ["cosine"]

    assert graph_client.get("/api/graph", params={"limit": 3}).json()["vertices"].__len__() == 3
    none = graph_client.get("/api/graph", params={"types": "PERSON"}).json()
    assert none["vertices"] == [] and none["edges"] == []
    assert graph_client.get("/api/graph", params={"entity": "zzz"}).json()["vertices"] == []


def test_graph_api_validation(graph_client: TestClient) -> None:
    bad = graph_client.get("/api/graph", params={"types": "concept,ALIEN"})
    assert bad.status_code == 422
    assert bad.json()["details"][0]["field"] == "types"
    assert graph_client.get("/api/graph", params={"hops": 4}).status_code == 422


def test_health_reports_graph_store(graph_client: TestClient) -> None:
    gs = graph_client.get("/api/health").json()["components"]["graph_store"]
    assert gs["status"] == "ok" and gs["info"]["backend"] == "networkx"
    assert int(gs["info"]["vertices"]) > 0


def test_q5_payload_fits_cap_with_heavily_cited_entities() -> None:
    """A real corpus has entities cited by hundreds of chunks; the Q5 event must stay intact."""
    import json

    from app.pipeline.stages import MAX_DATA_BYTES

    ids = [f"0a1b2c3d-v1-{i}" for i in range(300)]
    vertices = [
        Vertex(f"entity {i}", f"Entity {i}", "CONCEPT", "a" * 200, [f"E{i}"], ids, 5)
        for i in range(200)
    ]
    edges = [Edge(f"entity {i}", "relates to", f"entity {i + 1}", ids) for i in range(199)]
    sub = Subgraph(vertices, edges, {v.id: 1 for v in vertices})
    payload = {
        "subgraph": {
            "vertices": [v.summary() | {"hop": 1, "matched": False} for v in vertices],
            "edges": [e.summary() for e in edges],
        },
        "triples": triples(sub),
        "evidence": [{"chunk_id": c, "score": 1.0, "via": ["x"] * 5} for c in ids[:50]],
    }
    assert len(json.dumps(payload)) < MAX_DATA_BYTES
    assert "chunk_ids" not in payload["triples"][0]


def test_fusion_passes_at_most_top_k_to_rerank(graph_client: TestClient) -> None:
    stages, _, _ = run_query(
        graph_client,
        {"query": QUERY, "params": {"top_k": 2, "top_n": 2, "similarity_threshold": 0}},
    )
    rows = stages["Q6_fusion"]["data"]["rows"]
    assert len(rows) > 2  # vector + graph lists together exceed top_k
    assert [r["kept"] for r in rows] == [True, True] + [False] * (len(rows) - 2)
    assert len(stages["Q7_rerank"]["data"]["ranking"]) == 2


def test_graph_store_failure_falls_back_to_vector(app: FastAPI, graph_client: TestClient) -> None:
    class Down:
        backend = "neo4j"

        def stats(self):  # noqa: ANN201
            raise ConnectionError("Neo4j unreachable")

        def close(self) -> None:
            pass

    app.state.graph_store = Down()
    stages, _, done = run_query(
        graph_client, {"query": QUERY, "params": {"similarity_threshold": 0}}
    )
    q5 = stages["Q5_graph_retrieve"]
    assert q5["status"] == "warning" and "Neo4j unreachable" in q5["summary"]
    assert stages["Q6_fusion"]["data"] == {"skipped": True}
    assert stages["Q8_generate"]["data"]["llm_called"] is True
    assert done["status"] == "success"
