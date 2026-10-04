from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.graph.networkx_store import NetworkXGraphStore
from app.llm.base import LLMError
from app.pipeline.ingestion.kg_builder import build_graph, parse_extraction
from tests.helpers import FakeEmbedder, FakeLLM, make_text
from tests.test_ingest_api import ingest_ok


# ---------------------------------------------------------------- parsing
def test_parse_extraction_normalises_and_resolves_aliases() -> None:
    entities, relations = parse_extraction(
        {
            "entities": [
                {
                    "name": " Knowledge  Graph ",
                    "type": "concept",
                    "description": "x",
                    "aliases": ["KG"],
                },
                {"name": "Neo4j", "type": "database"},  # unknown type -> OTHER
                {"name": "!!!"},  # normalises to nothing -> dropped
            ],
            "relations": [
                {"source": "KG", "relation": "stored in", "target": "neo4j"},  # via alias + case
                {"source": "KG", "relation": "likes", "target": "Ghost"},  # unknown endpoint
                {"source": "Neo4j", "relation": "self", "target": "Neo4j"},  # self-loop
            ],
            "extra_field": "ignored",
        },
        "c1",
    )
    assert [(e.name, e.type, e.key) for e in entities] == [
        ("Knowledge Graph", "CONCEPT", "knowledge graph"),
        ("Neo4j", "OTHER", "neo4j"),
    ]
    assert entities[0].aliases == ["KG"]
    assert [(r.source, r.relation, r.target) for r in relations] == [
        ("knowledge graph", "stored in", "neo4j")
    ]


@pytest.mark.parametrize("bad", [{"entities": "nope"}, {"entities": [{"type": "X"}]}])
def test_parse_extraction_rejects_bad_schema(bad: dict) -> None:
    with pytest.raises(LLMError, match="schema"):
        parse_extraction(bad, "c1")


def test_build_graph_records_failures_and_embeds_names(tmp_path: Path) -> None:
    class Flaky(FakeLLM):
        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201
            if "BROKEN" in messages[-1]["content"]:
                raise LLMError("model returned garbage")
            return super().complete(messages, **kwargs)

    store, emb = NetworkXGraphStore(tmp_path), FakeEmbedder()
    chunks = [
        ("c1", "the vertex and edge of a graph traversal"),
        ("c2", "BROKEN passage"),
        ("c3", "embedding cosine similarity vector"),
    ]
    progress: list[tuple[int, int]] = []
    report = build_graph(store, emb, Flaky(), chunks, 42, lambda d, t: progress.append((d, t)))
    assert report.chunks_processed == 2
    assert report.failures == [{"chunk_id": "c2", "error": "model returned garbage"}]
    assert progress[-1] == (3, 3)
    assert store.stats()["vertices"] == report.added_vertices == 7
    assert store.keys_without_vectors() == []
    assert report.vectors_added == 7
    # a second build with the same chunks only merges
    again = build_graph(store, emb, FakeLLM(), chunks[:1], 42)
    assert (again.added_vertices, again.added_edges, again.vectors_added) == (0, 0, 0)


# ---------------------------------------------------------------- S8 in the ingest job
@pytest.fixture
def kg_client(app: FastAPI) -> Iterator[TestClient]:
    app.state.embedder = FakeEmbedder()
    app.state.llm_factory = lambda: FakeLLM()
    with TestClient(app) as c:
        yield c


def graph_chunk_ids(app: FastAPI) -> set[str]:
    sub = app.state.graph_store.overview(None, limit=10_000)
    return {cid for v in sub.vertices for cid in v.chunk_ids} | {
        cid for e in sub.edges for cid in e.chunk_ids
    }


def chroma_ids(app: FastAPI) -> set[str]:
    return set(app.state.vector_store._existing().get(include=[])["ids"])


def test_s8_builds_graph_with_provenance(app: FastAPI, kg_client: TestClient) -> None:
    _, stages, _ = ingest_ok(kg_client, [("notes.txt", make_text(seed=5, paragraphs=30).encode())])
    s8 = stages["S8_kg_build"]
    assert s8["status"] == "success", s8["summary"]
    data = s8["data"]
    assert data["vertex_count"] > 0 and data["edge_count"] > 0
    assert data["chunks_failed"] == 0
    assert data["types"] == {"CONCEPT": data["vertex_count"]}
    sample_vertex = data["sample"]["vertices"][0]
    assert "chunk_ids" not in sample_vertex and sample_vertex["chunk_count"] >= 1
    # provenance only ever points at stored, canonical chunks
    assert graph_chunk_ids(app) <= chroma_ids(app)
    # persisted to disk
    reloaded = NetworkXGraphStore(app.state.settings.data_dir)
    assert reloaded.stats()["vertices"] == data["vertex_count"]


def test_s8_handles_more_chunks_than_the_s7_preview(kg_client: TestClient) -> None:
    # S7 previews only its first 30 chunks; S8 must still read every chunk after S7 commits.
    content = make_text(seed=5, paragraphs=80).encode()
    _, stages, done = ingest_ok(kg_client, [("long.txt", content)], {"chunk_size": 128})
    assert stages["S4_chunk"]["data"]["chunk_count"] > 30
    assert done["status"] == "success", done
    assert stages["S8_kg_build"]["status"] == "success", stages["S8_kg_build"]["summary"]


def test_s8_rechunk_replaces_provenance(app: FastAPI, kg_client: TestClient) -> None:
    content = make_text(seed=5, paragraphs=30).encode()
    ingest_ok(kg_client, [("notes.txt", content)])
    before = graph_chunk_ids(app)
    _, stages, _ = ingest_ok(kg_client, [("notes.txt", content)], {"chunk_size": 256})
    s8 = stages["S8_kg_build"]["data"]
    after = graph_chunk_ids(app)
    assert after and not (after & before)  # all provenance moved to the new version's chunks
    assert after <= chroma_ids(app)
    assert s8["removed_vertices"] >= 0 and s8["removed_edges"] >= 0


def test_s8_warns_without_llm(app: FastAPI) -> None:
    app.state.embedder = FakeEmbedder()  # default llm_factory: no API key in tests
    with TestClient(app) as c:
        _, stages, done = ingest_ok(c, [("notes.txt", make_text(seed=5).encode())])
    s8 = stages["S8_kg_build"]
    assert s8["status"] == "warning"
    assert "LLM not configured" in s8["summary"]
    assert s8["data"]["vertex_count"] == 0
    assert done["status"] == "success"  # the documents are still searchable


def test_s8_skipped_when_build_graph_off(app: FastAPI, kg_client: TestClient) -> None:
    _, stages, _ = ingest_ok(
        kg_client, [("notes.txt", make_text(seed=5).encode())], {"build_graph": False}
    )
    s8 = stages["S8_kg_build"]
    assert s8["status"] == "success" and s8["data"]["skipped"] is True
    assert s8["summary"] == "build_graph is off"
    assert app.state.graph_store.stats()["vertices"] == 0
