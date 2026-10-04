from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.db.models import Run
from app.db.session import get_engine
from app.pipeline.ingestion.vector_store import Neighbour
from app.pipeline.query.runner import (
    NO_CONTEXT_ANSWER,
    QUERY_STAGES,
    chroma_where,
    collapse_and_filter,
    parse_citations,
)
from app.schemas.query import QueryFilters
from tests.helpers import FakeEmbedder, FakeLLM, FakeReranker, make_text, parse_sse
from tests.test_ingest_api import final_stages, ingest_ok

QUERY = "cosine similarity of embedding vectors"


# ---------------------------------------------------------------- units
def _hit(cid: str, sim: float, dup_of: str = "", doc: str = "d1") -> tuple[Neighbour, str]:
    meta = {"document_id": doc, "is_duplicate_of": dup_of, "cluster_id": 0}
    return Neighbour(cid, sim, meta), f"text of {cid}"


def test_collapse_merges_duplicates_into_canonical() -> None:
    hits = [_hit("dup1", 0.9, dup_of="canon"), _hit("canon", 0.88), _hit("other", 0.5)]
    canon = {"canon": ({"document_id": "d1", "is_duplicate_of": "", "cluster_id": 0}, "CANON")}
    out = collapse_and_filter(hits, canon, threshold=0.0)
    assert [c.chunk_id for c in out] == ["canon", "other"]
    assert out[0].text == "CANON"
    assert out[0].similarity == 0.9  # best similarity of the group
    assert out[0].collapsed_from == ["dup1"]


def test_collapse_keeps_duplicate_when_canonical_filtered_out() -> None:
    hits = [_hit("dup1", 0.9, dup_of="canon", doc="d1")]
    canon = {"canon": ({"document_id": "d2", "is_duplicate_of": "", "cluster_id": 0}, "CANON")}
    out = collapse_and_filter(hits, canon, 0.0, QueryFilters(document_ids=["d1"]))
    assert [c.chunk_id for c in out] == ["dup1"]
    assert out[0].collapsed_from == []


def test_threshold_marks_candidates_filtered() -> None:
    out = collapse_and_filter([_hit("a", 0.6), _hit("b", 0.2)], {}, threshold=0.35)
    assert out[0].filtered is None
    assert out[1].filtered == "similarity 0.20 < threshold 0.35"


def test_parse_citations() -> None:
    valid = {"0a1b2c3d-v1-0", "0a1b2c3d-v1-1"}
    answer = "X [0a1b2c3d-v1-1]. Y [0a1b2c3d-v1-0, 0a1b2c3d-v1-1] Z [ffffffff-v1-9] [note]"
    cited, unknown = parse_citations(answer, valid)
    assert cited == ["0a1b2c3d-v1-1", "0a1b2c3d-v1-0"]
    assert unknown == ["ffffffff-v1-9"]


def test_chroma_where() -> None:
    assert chroma_where(None) is None
    assert chroma_where(QueryFilters(document_ids=["d"])) == {"document_id": {"$in": ["d"]}}
    both = chroma_where(QueryFilters(document_ids=["d"], cluster_ids=[1]))
    assert both == {"$and": [{"document_id": {"$in": ["d"]}}, {"cluster_id": {"$in": [1]}}]}


# ---------------------------------------------------------------- API
@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def query_client(app: FastAPI, llm: FakeLLM) -> Iterator[TestClient]:
    app.state.embedder = FakeEmbedder()
    app.state.reranker = FakeReranker()
    app.state.llm_factory = lambda: llm
    with TestClient(app) as c:
        yield c


def _ingest(client: TestClient) -> None:
    ingest_ok(client, [("notes.txt", make_text(seed=3, paragraphs=40).encode())])


def run_query(client: TestClient, body: dict) -> tuple[dict[str, dict], list[dict], dict]:
    resp = client.post("/api/query", json=body)
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["run_id"]
    events = parse_sse(client.get(f"/api/query/{run_id}/events").text)
    assert events[-1]["event"] == "done"
    return final_stages(events), events, events[-1]["data"]


def test_query_end_to_end(query_client: TestClient, llm: FakeLLM) -> None:
    _ingest(query_client)
    stages, events, done = run_query(
        query_client, {"query": QUERY, "params": {"similarity_threshold": 0.0, "top_n": 2}}
    )
    assert list(stages) == list(QUERY_STAGES)
    # Q2 may warn off-topic: the hashed test embedder gives low topic similarity
    assert stages["Q2_input_guardrail"]["status"] in ("success", "warning")
    # Q10 warns: FakeLLM cannot act as a judge (eval judges are covered in test_eval.py)
    others = {k: s for k, s in stages.items() if k not in ("Q2_input_guardrail", "Q10_eval")}
    assert all(s["status"] == "success" for s in others.values()), {
        k: (s["status"], s["summary"]) for k, s in others.items()
    }
    assert stages["Q10_eval"]["status"] == "warning"
    assert done["eval"]["metrics"]["answer_relevancy"]["value"] is not None

    assert stages["Q3_query_embed"]["data"]["token_count"] > 0
    q4 = stages["Q4_vector_retrieve"]
    assert q4["params_used"]["top_k"] == 10
    sims = [c["similarity"] for c in q4["data"]["candidates"]]
    assert sims == sorted(sims, reverse=True)

    ranking = stages["Q7_rerank"]["data"]["ranking"]
    assert [r["after_rank"] for r in ranking] == list(range(1, len(ranking) + 1))
    assert sum(r["kept"] for r in ranking) == 2

    q8 = stages["Q8_generate"]["data"]
    kept_ids = [r["chunk_id"] for r in ranking if r["kept"]]
    assert q8["context_chunk_ids"] == kept_ids
    assert all(f"[{cid}]" in q8["prompt"][1]["content"] for cid in kept_ids)
    assert q8["prompt"][1]["content"].endswith(f"Question: {QUERY}")

    tokens = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert tokens == q8["answer"] == done["answer"]
    assert done["status"] == "success"
    assert [c["chunk_id"] for c in done["citations"]] == [kept_ids[0]]
    assert done["citations"][0]["filename"] == "notes.txt"
    assert len(llm.calls) == 1

    # token events arrive between Q8 running and Q8 terminal
    order = [(e["event"], e["data"].get("stage_id"), e["data"].get("status")) for e in events]
    first_token = next(i for i, o in enumerate(order) if o[0] == "token")
    assert order.index(("stage", "Q8_generate", "running")) < first_token
    assert first_token < order.index(("stage", "Q8_generate", "success"))


def test_threshold_filters_everything_without_calling_llm(
    query_client: TestClient, llm: FakeLLM
) -> None:
    _ingest(query_client)
    stages, _, done = run_query(
        # graph_hops=0: vector-only, so nothing reaches the LLM (graph evidence skips the threshold)
        query_client,
        {"query": QUERY, "params": {"similarity_threshold": 1.0, "graph_hops": 0}},
    )
    q4 = stages["Q4_vector_retrieve"]
    assert q4["status"] == "warning"
    assert q4["data"]["kept"] == 0
    assert all(c["filtered"].startswith("similarity") for c in q4["data"]["candidates"])
    assert stages["Q7_rerank"]["data"] == {"skipped": True}
    assert stages["Q8_generate"]["status"] == "warning"
    assert done["answer"] == NO_CONTEXT_ANSWER
    assert llm.calls == []


def test_empty_collection(query_client: TestClient, llm: FakeLLM) -> None:
    stages, _, done = run_query(query_client, {"query": QUERY})
    assert stages["Q4_vector_retrieve"]["status"] == "warning"
    assert done["status"] == "success"
    assert done["answer"] == NO_CONTEXT_ANSWER
    assert llm.calls == []


def test_document_filter(query_client: TestClient) -> None:
    _ingest(query_client)
    stages, _, _ = run_query(
        query_client,
        {
            "query": QUERY,
            "filters": {"document_ids": ["nope"]},
            "params": {"similarity_threshold": 0},
        },
    )
    assert stages["Q4_vector_retrieve"]["data"]["fetched"] == 0


def test_unknown_citation_is_a_warning(app: FastAPI, query_client: TestClient) -> None:
    app.state.llm_factory = lambda: FakeLLM(extra=" Also [ffffffff-v1-0].")
    _ingest(query_client)
    stages, _, done = run_query(
        query_client, {"query": QUERY, "params": {"similarity_threshold": 0}}
    )
    q8 = stages["Q8_generate"]
    assert q8["status"] == "warning"
    assert q8["data"]["unknown_citations"] == ["ffffffff-v1-0"]
    assert done["status"] == "success"


def test_llm_not_configured_fails_q8_only(app: FastAPI, query_client: TestClient) -> None:
    from app.llm.factory import build_llm

    app.state.llm_factory = lambda: build_llm(app.state.settings)  # no API key in tests
    _ingest(query_client)
    stages, _, done = run_query(
        query_client, {"query": QUERY, "params": {"similarity_threshold": 0}}
    )
    assert stages["Q4_vector_retrieve"]["status"] == "success"
    assert stages["Q7_rerank"]["status"] == "success"
    assert stages["Q8_generate"]["status"] == "error"
    assert "OPENAI_API_KEY" in stages["Q8_generate"]["summary"]
    assert stages["Q10_eval"]["status"] == "pending"  # never reached
    assert done["status"] == "error"
    with Session(get_engine()) as s:
        assert s.get(Run, stages["Q1_validate"]["run_id"]).status == "error"


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"query": "hi"}, "query"),
        ({"query": QUERY, "params": {"top_k": 2, "top_n": 3}}, "params.top_n"),
        ({"query": QUERY, "params": {"top_k": 2}}, "params.top_n"),  # default top_n=4 > 2
        ({"query": QUERY, "collection": "other_col"}, "collection"),
        ({"query": QUERY, "surprise": 1}, "surprise"),
    ],
)
def test_query_validation_errors(query_client: TestClient, body: dict, field: str) -> None:
    resp = query_client.post("/api/query", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"] == "VALIDATION_ERROR"
    assert field in [d["field"] for d in resp.json()["details"]], resp.json()


def test_query_events_replay_from_db(query_client: TestClient) -> None:
    from app.core.events import bus

    stages, _, _ = run_query(query_client, {"query": QUERY})
    run_id = stages["Q1_validate"]["run_id"]
    bus._channels.pop(run_id)  # simulate eviction / restart
    events = parse_sse(query_client.get(f"/api/query/{run_id}/events").text)
    assert events[-1]["data"]["replayed"] is True
    assert events[-1]["data"]["answer"] == NO_CONTEXT_ANSWER


def test_same_query_same_order(query_client: TestClient) -> None:
    _ingest(query_client)
    body = {"query": QUERY, "params": {"similarity_threshold": 0, "seed": 7}}
    orders = []
    for _ in range(2):
        stages, _, _ = run_query(query_client, body)
        q4 = [c["chunk_id"] for c in stages["Q4_vector_retrieve"]["data"]["candidates"]]
        q7 = [r["chunk_id"] for r in stages["Q7_rerank"]["data"]["ranking"]]
        orders.append((q4, q7))
    assert orders[0] == orders[1]


def test_q8_payload_fits_cap_at_slider_maximums() -> None:
    import json

    from app.pipeline.query.runner import Candidate, build_messages
    from app.pipeline.stages import MAX_DATA_BYTES

    text = make_text(seed=1, paragraphs=200)[: 2048 * 5]  # ~2048 tokens of English prose
    context = [
        Candidate(f"0a1b2c3d-v1-{i}", text, 0.5, {"filename": "f.txt", "page": 1, "page_end": 1})
        for i in range(20)  # top_n max
    ]
    data = {
        "prompt": build_messages(QUERY, context),
        "answer": "x" * 6000,
        "citations": [c.view() for c in context],
    }
    assert len(json.dumps(data)) < MAX_DATA_BYTES


# ---------------------------------------------------------------- OpenAI streaming (mocked HTTP)
def _openai_stream(handler):  # noqa: ANN001, ANN202
    import httpx

    from app.llm.openai_provider import OpenAIProvider

    return OpenAIProvider(
        "sk-test", "m", "m-fast", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_openai_stream_emits_deltas_and_usage() -> None:
    import json

    import httpx

    seen: dict = {}

    def chunk(**fields) -> str:  # noqa: ANN003
        base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m-2026"}
        return "data: " + json.dumps(base | fields) + "\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = (
            chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}])
            + chunk(choices=[{"index": 0, "delta": {"content": "lo [x]"}}])
            + chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
            + chunk(
                choices=[],
                usage={"prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53},
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    deltas: list[str] = []
    result = _openai_stream(handler).stream(
        [{"role": "user", "content": "q"}], deltas.append, temperature=0.2, seed=42, max_tokens=99
    )
    assert deltas == ["Hel", "lo [x]"]
    assert (result.text, result.model) == ("Hello [x]", "m-2026")
    assert (result.prompt_tokens, result.completion_tokens) == (50, 3)
    assert seen["stream"] is True
    assert seen["stream_options"] == {"include_usage": True}
    assert (seen["seed"], seen["max_completion_tokens"], seen["temperature"]) == (42, 99, 0.2)


def test_openai_stream_error_is_llm_error() -> None:
    import httpx

    from app.llm.base import LLMError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key", "type": "auth"}})

    with pytest.raises(LLMError, match="OpenAI request failed"):
        _openai_stream(handler).stream([{"role": "user", "content": "q"}], lambda _: None)


def test_run_history(query_client: TestClient) -> None:
    _ingest(query_client)
    run_query(query_client, {"query": QUERY})
    runs = query_client.get("/api/runs").json()
    assert runs["total"] == 2
    assert [r["kind"] for r in runs["items"]] == ["query", "ingest"]  # newest first
    assert runs["items"][0]["label"] == QUERY
    assert runs["items"][0]["status"] == "success"
    assert runs["items"][1]["label"] == "notes.txt"
    only = query_client.get("/api/runs", params={"kind": "ingest"}).json()
    assert [r["kind"] for r in only["items"]] == ["ingest"]
    assert query_client.get("/api/runs", params={"kind": "bogus"}).status_code == 422
