from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, col, select

from app.db.models import Chunk, Cluster, Document
from app.db.session import get_engine
from app.pipeline.ingestion.runner import INGEST_LOCK
from tests.helpers import FakeEmbedder, FakeLLM, make_text
from tests.test_ingest_api import assert_store_consistent, ingest_ok
from tests.test_kg_build import chroma_ids, graph_chunk_ids

TEXT_A = make_text(seed=1, paragraphs=30).encode()
TEXT_B = make_text(seed=9, paragraphs=12, topic_cycle=["cooking", "finance"]).encode()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    app.state.embedder = FakeEmbedder()
    app.state.llm_factory = lambda: FakeLLM()
    with TestClient(app) as c:
        yield c


def doc_id(stages: dict) -> str:
    return stages["S1_upload"]["data"]["files"][0]["document_id"]


def chunks(**where) -> list[Chunk]:
    with Session(get_engine()) as s:
        stmt = select(Chunk)
        for k, v in where.items():
            stmt = stmt.where(getattr(Chunk, k) == v)
        return list(s.exec(stmt).all())


def test_delete_removes_document_everywhere(app: FastAPI, client: TestClient) -> None:
    _, sa, _ = ingest_ok(client, [("a.txt", TEXT_A)])
    _, sb, _ = ingest_ok(client, [("b.txt", TEXT_B)])
    a, b = doc_id(sa), doc_id(sb)
    a_chunks = {c.id for c in chunks(document_id=a)}
    assert a_chunks & graph_chunk_ids(app)

    resp = client.delete(f"/api/documents/{a}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "a.txt" and body["chunks_deleted"] == len(a_chunks)
    assert body["vectors_deleted"] == len(a_chunks) and body["duplicates_promoted"] == 0

    assert [d["id"] for d in client.get("/api/documents").json()] == [b]
    assert chunks(document_id=a) == []
    assert not (a_chunks & chroma_ids(app)) and not (a_chunks & graph_chunk_ids(app))
    assert graph_chunk_ids(app) <= chroma_ids(app)
    assert_store_consistent(app)
    with Session(get_engine()) as s:  # cluster sizes match the remaining canonical chunks
        for cl in s.exec(select(Cluster)).all():
            canon = s.exec(
                select(Chunk.id).where(
                    Chunk.cluster_id == cl.id, col(Chunk.is_duplicate_of).is_(None)
                )
            ).all()
            assert cl.size == len(canon) > 0

    # the same file can be uploaded again (its hash no longer counts as ingested)
    _, again, _ = ingest_ok(client, [("a.txt", TEXT_A)])
    assert again["S1_upload"]["data"]["files"][0]["action"] == "new"


def test_delete_promotes_duplicates_and_moves_graph_facts(app: FastAPI, client: TestClient) -> None:
    _, sa, _ = ingest_ok(client, [("a.txt", TEXT_A)])
    _, sb, _ = ingest_ok(client, [("b.txt", TEXT_A + b"\n\nOne extra closing line.")])
    _, sc, _ = ingest_ok(client, [("c.txt", TEXT_A + b"\n\nA different closing line.")])
    a, b, c = doc_id(sa), doc_id(sb), doc_id(sc)
    dups = [x for x in chunks() if x.is_duplicate_of and x.document_id in (b, c)]
    assert dups and {d.is_duplicate_of for d in dups} <= {x.id for x in chunks(document_id=a)}
    canon = dups[0].is_duplicate_of
    group = sorted(d.id for d in dups if d.is_duplicate_of == canon)
    facts_before = graph_chunk_ids(app)
    assert canon in facts_before

    body = client.delete(f"/api/documents/{a}").json()
    assert body["duplicates_promoted"] == len({d.is_duplicate_of for d in dups})
    assert body["duplicates_repointed"] == len(dups) - body["duplicates_promoted"]

    heir, *rest = group
    by_id = {x.id: x for x in chunks()}
    assert by_id[heir].is_duplicate_of is None and by_id[heir].duplicate_similarity is None
    assert all(by_id[r].is_duplicate_of == heir for r in rest)
    facts_after = graph_chunk_ids(app)
    assert heir in facts_after and canon not in facts_after  # provenance moved, not lost
    assert facts_after <= chroma_ids(app)
    assert_store_consistent(app)
    with Session(get_engine()) as s:
        for d in (b, c):
            doc = s.get(Document, d)
            assert doc.duplicate_chunk_count == sum(
                x.is_duplicate_of is not None for x in chunks(document_id=d)
            )


RAW_PII = ["ABCPM1234K", "2345 6789 0124", "ravi.menon@example.com", "98765 43210"]


def test_chunk_text_is_served_with_pii_masked(client: TestClient) -> None:
    text = (
        "Policyholder: Ravi Menon. PAN: ABCPM1234K. Aadhaar: 2345 6789 0124. "
        "Email: ravi.menon@example.com. Phone: +91 98765 43210.\n\n"
    ) + make_text(seed=3, paragraphs=4)
    ingest_ok(client, [("pii.txt", text.encode())])
    first = next(c for c in chunks() if "ABCPM1234K" in c.text)

    one = client.get(f"/api/chunks/{first.id}")  # the citation popup
    listed = client.get("/api/chunks")  # the Documents page
    assert one.status_code == listed.status_code == 200
    served = one.text + listed.text
    assert not [v for v in RAW_PII if v in served]
    for placeholder in ("<IN_PAN>", "<IN_AADHAAR>", "<EMAIL_ADDRESS>", "<PHONE_NUMBER>"):
        assert placeholder in one.json()["text"]
    assert "ABCPM1234K" in next(c for c in chunks() if c.id == first.id).text  # store unchanged

    cfg = client.get("/api/config").json()
    cfg["guardrails"]["output"]["pii_leak"]["enabled"] = False
    assert client.put("/api/config", json=cfg).status_code == 200
    assert "ABCPM1234K" in client.get(f"/api/chunks/{first.id}").json()["text"]


def test_delete_unknown_and_while_ingesting(client: TestClient) -> None:
    resp = client.delete("/api/documents/nope")
    assert resp.status_code == 404 and resp.json()["error"] == "NOT_FOUND"
    _, sa, _ = ingest_ok(client, [("a.txt", TEXT_A)])
    with INGEST_LOCK:  # an ingest job holds the lock
        resp = client.delete(f"/api/documents/{doc_id(sa)}")
    assert resp.status_code == 409 and resp.json()["error"] == "CONFLICT"
    assert len(client.get("/api/documents").json()) == 1
