import json
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.events import bus
from app.db.models import Chunk
from app.db.session import get_engine
from app.pipeline.ingestion.runner import INGEST_STAGES
from tests.helpers import FakeEmbedder, make_pdf, make_text, parse_sse


def assert_store_consistent(app: FastAPI) -> None:
    """SQLite and Chroma hold the same chunks with matching links; links resolve to canonicals."""
    with Session(get_engine()) as s:
        rows = {c.id: c for c in s.exec(select(Chunk)).all()}
    vs = app.state.vector_store
    info = vs.collection_info()
    if not info["exists"]:
        assert rows == {}
        return
    col = vs._existing()
    got = col.get(include=["metadatas"])
    meta = dict(zip(got["ids"], got["metadatas"], strict=True))
    assert set(meta) == set(rows)
    for cid, row in rows.items():
        assert meta[cid]["is_duplicate_of"] == (row.is_duplicate_of or "")
        assert meta[cid]["cluster_id"] == row.cluster_id
        assert row.cluster_id >= 0, cid
        if row.is_duplicate_of is not None:
            target = rows.get(row.is_duplicate_of)
            assert target is not None, f"{cid} -> missing {row.is_duplicate_of}"
            assert target.is_duplicate_of is None, f"{cid} -> non-root {target.id}"
            assert target.cluster_id == row.cluster_id


@pytest.fixture
def ingest_client(app: FastAPI) -> Iterator[TestClient]:
    app.state.embedder = FakeEmbedder()
    with TestClient(app) as c:
        yield c


def upload(client: TestClient, files: list[tuple[str, bytes]], params: dict | None = None):
    data = {"params": json.dumps(params)} if params is not None else None
    return client.post(
        "/api/ingest",
        files=[("files", (name, content)) for name, content in files],
        data=data,
    )


def events_for(client: TestClient, job_id: str, last_id: int | None = None) -> list[dict]:
    headers = {"Last-Event-ID": str(last_id)} if last_id is not None else {}
    resp = client.get(f"/api/ingest/{job_id}/events", headers=headers)
    assert resp.status_code == 200, resp.text
    return parse_sse(resp.text)


def final_stages(events: list[dict]) -> dict[str, dict]:
    """Last event per stage (terminal status)."""
    out: dict[str, dict] = {}
    for e in events:
        if e["event"] == "stage":
            out[e["data"]["stage_id"]] = e["data"]
    return out


def ingest_ok(client: TestClient, files, params=None) -> tuple[str, dict[str, dict], dict]:
    resp = upload(client, files, params)
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    events = events_for(client, job_id)
    assert events[-1]["event"] == "done"
    return job_id, final_stages(events), events[-1]["data"]


TEXT_A = make_text(seed=1, paragraphs=30).encode()


def test_ingest_txt_runs_all_stages_with_real_counts(ingest_client: TestClient):
    job_id, stages, done = ingest_ok(ingest_client, [("a.txt", TEXT_A)])
    assert list(stages) == list(INGEST_STAGES)
    assert all(s["status"] in ("success", "warning") for s in stages.values()), stages
    assert done["status"] == "success"

    s4 = stages["S4_chunk"]
    assert s4["params_used"] == {"chunk_size": 512, "chunk_overlap": 64}
    n_chunks = s4["data"]["chunk_count"]
    assert n_chunks > 1 and sum(b["count"] for b in s4["data"]["histogram"]) == n_chunks
    assert stages["S3_tokenize"]["data"]["total_tokens"] > 0
    assert len(stages["S3_tokenize"]["data"]["preview"]["tokens"]) == 300
    assert stages["S5_embed"]["data"]["dimension"] == 64
    assert len(stages["S5_embed"]["data"]["points"]) == n_chunks
    assert stages["S6_segregate"]["status"] == "warning"  # keyword labels: no LLM key in tests
    assert stages["S6_segregate"]["data"]["clusters"]
    assert stages["S7_vector_store"]["data"]["total_vectors"] == n_chunks
    for s in stages.values():
        assert s["duration_ms"] is not None

    docs = ingest_client.get("/api/documents").json()
    assert len(docs) == 1 and docs[0]["status"] == "ready" and docs[0]["chunk_count"] == n_chunks
    page = ingest_client.get("/api/chunks", params={"document_id": docs[0]["id"]}).json()
    assert page["total"] == n_chunks
    assert page["items"][0]["id"] == f"{docs[0]['id'][:8]}-v1-0"
    assert_store_consistent(ingest_client.app)
    assert ingest_client.get("/api/health").json()["components"]["chroma"]["info"][
        "vectors"
    ] == str(n_chunks)


def test_same_file_twice_is_skipped(ingest_client: TestClient):
    ingest_ok(ingest_client, [("a.txt", TEXT_A)])
    _, stages, _ = ingest_ok(ingest_client, [("a-copy.txt", TEXT_A)])
    assert stages["S1_upload"]["data"]["files"][0]["action"] == "duplicate"
    assert stages["S4_chunk"]["data"] == {"skipped": True}
    assert len(ingest_client.get("/api/documents").json()) == 1


def test_reingest_with_new_chunk_size_replaces_chunks(ingest_client: TestClient, app: FastAPI):
    _, first, _ = ingest_ok(ingest_client, [("a.txt", TEXT_A)], {"chunk_size": 512})
    _, second, _ = ingest_ok(
        ingest_client, [("a.txt", TEXT_A)], {"chunk_size": 128, "chunk_overlap": 16}
    )
    assert second["S1_upload"]["data"]["files"][0]["action"] == "rechunk"
    n1, n2 = first["S4_chunk"]["data"]["chunk_count"], second["S4_chunk"]["data"]["chunk_count"]
    assert n2 > n1
    assert second["S7_vector_store"]["data"]["deleted_previous_version"] == n1
    assert second["S7_vector_store"]["data"]["total_vectors"] == n2
    # Old version must not be treated as duplicates of the new one (within-batch matches can
    # occur: the fake embedder's tiny vocabulary makes same-topic chunks near-identical).
    dups = second["S6_segregate"]["data"]["duplicates"]
    assert all(d["source"] == "batch" and "-v2-" in d["duplicate_of"] for d in dups)

    [doc] = ingest_client.get("/api/documents").json()
    assert doc["version"] == 2 and doc["chunk_count"] == n2 and doc["chunk_size"] == 128
    ids = [c["id"] for c in ingest_client.get("/api/chunks", params={"limit": 200}).json()["items"]]
    assert ids and all("-v2-" in i for i in ids)
    assert app.state.vector_store.count() == n2
    assert_store_consistent(app)


def test_near_duplicate_chunks_are_linked_to_canonical(ingest_client: TestClient):
    ingest_ok(ingest_client, [("a.txt", TEXT_A)])
    # Different file (new hash) whose content repeats document A.
    _, stages, _ = ingest_ok(ingest_client, [("b.txt", TEXT_A + b"\n\nOne extra closing line.")])
    s6 = stages["S6_segregate"]["data"]
    assert s6["duplicate_count"] >= 1
    first = s6["duplicates"][0]
    assert first["source"] == "existing" and first["similarity"] >= 0.95
    assert first["duplicate_of_text"]
    doc_b = stages["S1_upload"]["data"]["files"][0]["document_id"]
    dups = ingest_client.get("/api/chunks", params={"duplicates": True, "document_id": doc_b})
    assert dups.json()["total"] == s6["duplicate_count"]
    assert_store_consistent(ingest_client.app)


def test_pdf_ingest_reports_pages_and_removes_headers(ingest_client: TestClient):
    pages = [make_text(seed=i, paragraphs=3) for i in range(5)]
    pdf = make_pdf(pages, header="ACME Confidential Report")
    _, stages, _ = ingest_ok(
        ingest_client, [("report.pdf", pdf)], {"chunk_size": 128, "chunk_overlap": 16}
    )
    assert stages["S1_upload"]["data"]["files"][0]["page_count"] == 5
    s2 = stages["S2_parse"]["data"]["documents"][0]
    assert any("ACME" in line for line in s2["removed_header_footer_lines"])
    assert "ACME" not in s2["clean_preview"]
    chunks = stages["S4_chunk"]["data"]["chunks"]
    assert chunks[0]["page"] == 1 and max(c["page_end"] for c in chunks) > 1


def test_sse_resume_204_and_db_replay(ingest_client: TestClient):
    job_id, _, _ = ingest_ok(ingest_client, [("a.txt", TEXT_A)])
    events = events_for(ingest_client, job_id)
    done_id = int(events[-1]["id"])
    assert [int(e["id"]) for e in events] == list(range(1, done_id + 1))

    resumed = events_for(ingest_client, job_id, last_id=done_id - 2)
    assert [int(e["id"]) for e in resumed] == [done_id - 1, done_id]
    caught_up = ingest_client.get(
        f"/api/ingest/{job_id}/events", headers={"Last-Event-ID": str(done_id)}
    )
    assert caught_up.status_code == 204

    bus._channels.pop(job_id)  # simulate eviction / server restart
    replayed = events_for(ingest_client, job_id)
    assert replayed[-1]["event"] == "done" and replayed[-1]["data"]["replayed"] is True
    assert final_stages(replayed).keys() == final_stages(events).keys()
    assert ingest_client.get("/api/ingest/nope/events").status_code == 404


def test_failing_embedder_ends_job_with_error_and_releases_lock(
    ingest_client: TestClient, app: FastAPI
):
    app.state.embedder = FakeEmbedder(fail=True)
    _, stages, done = ingest_ok(ingest_client, [("a.txt", TEXT_A)])
    assert stages["S5_embed"]["status"] == "error"
    assert "exploded" in stages["S5_embed"]["summary"]
    assert done["status"] == "error"
    assert ingest_client.get("/api/documents").json()[0]["status"] == "error"

    [failed_doc] = ingest_client.get("/api/documents").json()

    app.state.embedder = FakeEmbedder()
    _, stages, done = ingest_ok(ingest_client, [("a.txt", TEXT_A)])
    assert done["status"] == "success"
    retry = stages["S1_upload"]["data"]["files"][0]
    assert retry["action"] == "new" and retry["document_id"] == failed_doc["id"]
    [doc] = ingest_client.get("/api/documents").json()  # error row reused, not duplicated
    assert doc["status"] == "ready" and doc["error"] is None
    assert_store_consistent(app)


@pytest.mark.parametrize(
    "files, params, fields",
    [
        ([("a.exe", b"MZ")], None, {"files.0"}),
        ([("a.pdf", b"plain text")], None, {"files.0"}),
        ([("a.docx", b"%PDF-1.4")], None, {"files.0"}),
        ([("a.txt", b"x\x00y")], None, {"files.0"}),
        ([(f"f{i}.txt", b"hi") for i in range(11)], None, {"files"}),
        ([("a.txt", b"hello")], "not-json", {"params"}),
        (
            [("a.txt", b"hello")],
            {"chunk_size": 256, "chunk_overlap": 200},
            {"params.chunk_overlap"},
        ),
        ([("a.txt", b"hello")], {"chunk_overlap": 300}, {"params.chunk_overlap"}),
        ([("a.txt", b"hello")], {"top_k": 3}, {"params.top_k"}),
        ([("a.txt", b"hello")], {"dedup_threshold": 0.5}, {"params.dedup_threshold"}),
    ],
)
def test_ingest_validation_rejected_before_any_stage(ingest_client, files, params, fields):
    data = None
    if params is not None:
        data = {"params": params if isinstance(params, str) else json.dumps(params)}
    resp = ingest_client.post("/api/ingest", files=[("files", (n, c)) for n, c in files], data=data)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"] == "VALIDATION_ERROR"
    assert {d["field"] for d in body["details"]} == fields
    assert ingest_client.get("/api/documents").json() == []


def test_ingest_without_files(ingest_client: TestClient):
    resp = ingest_client.post("/api/ingest", data={"params": "{}"})
    assert resp.status_code == 422
    assert resp.json()["details"][0]["field"] == "files"


def test_rechunk_of_canonical_doc_keeps_duplicate_links_valid(
    ingest_client: TestClient, app: FastAPI
):
    # A, then B (repeats A -> B's chunks are duplicates of A-v1), then re-chunk A.
    ingest_ok(ingest_client, [("a.txt", TEXT_A)])
    _, b_stages, _ = ingest_ok(ingest_client, [("b.txt", TEXT_A + b"\n\nOne extra closing line.")])
    assert b_stages["S6_segregate"]["data"]["duplicate_count"] >= 1
    assert_store_consistent(app)

    # Same size, different overlap: A-v2 chunks are near-identical to B's chunks.
    _, stages, _ = ingest_ok(ingest_client, [("a.txt", TEXT_A)], {"chunk_overlap": 32})
    assert stages["S1_upload"]["data"]["files"][0]["action"] == "rechunk"
    assert stages["S6_segregate"]["data"]["promoted_orphans"] >= 1
    assert_store_consistent(app)

    # Nothing links to A's deleted v1 chunks any more.
    doc_a = stages["S1_upload"]["data"]["files"][0]["document_id"]
    with Session(get_engine()) as s:
        links = [c.is_duplicate_of for c in s.exec(select(Chunk)).all() if c.is_duplicate_of]
    assert not any(link.startswith(f"{doc_a[:8]}-v1-") for link in links)
