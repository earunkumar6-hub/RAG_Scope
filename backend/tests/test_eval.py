import json
import re
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.eval import judge
from app.eval.metrics import (
    hit_rate,
    reciprocal_rank,
    relevance_flags,
    spans_match,
    strip_citations,
    summarise,
)
from app.eval.runner import expand_grid
from app.guardrails.output_guards import NOT_FOUND
from app.llm.base import LLMError, LLMResult
from app.llm.usage import MeteredLLM, UsageMeter, active_meter
from tests.helpers import FakeEmbedder, FakeLLM, FakeReranker, make_text, parse_sse
from tests.test_ingest_api import final_stages, ingest_ok

QUERY = "cosine similarity of embedding vectors"


def span(start: int, end: int, sha: str = "a") -> dict:
    return {"sha256": sha, "start": start, "end": end}


# ---------------------------------------------------------------- deterministic metrics
def test_spans_match_needs_half_of_the_shorter_span() -> None:
    assert spans_match(span(0, 100), span(50, 150))  # 50 / 100
    assert not spans_match(span(0, 100), span(51, 151))
    assert spans_match(span(0, 1000), span(100, 200))  # small span fully inside a large one
    assert not spans_match(span(0, 100), span(0, 100, sha="b"))
    assert not spans_match(span(0, 100), span(100, 200))  # touching, no overlap


def test_relevance_flags_hit_rate_and_mrr() -> None:
    relevant = [span(500, 600)]
    retrieved = [[span(0, 100)], [span(200, 300), span(510, 590)], [span(500, 600)]]
    flags = relevance_flags(retrieved, relevant)
    assert flags == [False, True, True]  # item 2 matches through its collapsed duplicate
    assert hit_rate(flags) == 1.0
    assert reciprocal_rank(flags) == 0.5
    assert hit_rate([False]) == reciprocal_rank([False]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_summarise_ignores_missing_values() -> None:
    out = summarise([{"a": 1.0, "b": None}, {"a": 0.0, "b": None}, {"a": 0.5}])
    assert out == {"a": {"mean": 0.5, "n": 3}, "b": {"mean": None, "n": 0}}


def test_expand_grid() -> None:
    assert expand_grid({}) == [{}]
    cells = expand_grid({"top_k": [5, 10], "top_n": [2, 3]})
    assert cells == [
        {"top_k": 5, "top_n": 2},
        {"top_k": 5, "top_n": 3},
        {"top_k": 10, "top_n": 2},
        {"top_k": 10, "top_n": 3},
    ]


def test_strip_citations() -> None:
    assert strip_citations("Graphs [0a1b2c3d-v1-4]. Edges [x, y].") == "Graphs. Edges."


# ---------------------------------------------------------------- usage meter
def test_meter_records_tokens_per_stage_and_cache_hits() -> None:
    stage = {"now": "Q5"}
    meter = UsageMeter(lambda: stage["now"])
    llm = MeteredLLM(JudgeLLM(), meter)
    entities = "List the entities mentioned in the question."
    llm.complete([{"role": "system", "content": entities}, {"role": "user", "content": "graphs"}])
    stage["now"] = "Q8"
    llm.stream(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "[0a1b2c3d-v1-0] t"}],
        lambda _: None,
    )
    token = active_meter.set(meter)
    try:
        meter.cache_hit()
    finally:
        active_meter.reset(token)
    summary = meter.summary()
    assert summary["by_stage"]["Q5"] == {
        "calls": 1,
        "cache_hits": 0,
        "prompt_tokens": 50,
        "completion_tokens": 20,
    }
    assert summary["by_stage"]["Q8"]["calls"] == 1
    assert summary["by_stage"]["Q8"]["cache_hits"] == 1
    assert summary["total"]["prompt_tokens"] == 150


# ---------------------------------------------------------------- judges
class JudgeLLM(FakeLLM):
    """FakeLLM that also plays every judge and the synthetic Q&A writer, deterministically."""

    def __init__(self, extra: str = ""):
        super().__init__(extra)
        self.kinds: list[str] = []

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201
        system, user = messages[0]["content"], messages[-1]["content"]
        if "is supported by its context" in system:
            answer = user.split("Answer:\n", 1)[1]
            out = {"sentences": [{"text": s, "supported": True} for s in answer.split(". ") if s]}
            kind = "groundedness"
        elif "judge context passages" in system:
            n = len(re.findall(r"^\[\d+\] ", user, re.M))
            out = {
                "passages": [
                    {"index": i, "relevant": i % 2 == 1, "reason": f"r{i}"} for i in range(1, n + 1)
                ]
            }
            kind = "precision"
        elif "reference answer is covered" in system:
            out = {
                "statements": [
                    {"text": "s1", "attributed": True, "reason": ""},
                    {"text": "s2", "attributed": False, "reason": ""},
                ]
            }
            kind = "recall"
        elif "compare an answer with a reference" in system:
            out = {
                "reference": [
                    {"statement": "a", "covered": True},
                    {"statement": "b", "covered": False},
                ],
                "answer": [
                    {"statement": "a", "verdict": "supported"},
                    {"statement": "x", "verdict": "extra"},  # neutral: in no group
                ],
            }
            kind = "correctness"
        elif "write evaluation data" in system:
            words = re.findall(r"[a-z]+", user.lower())[1:5]
            out = {"question": "What about " + " ".join(words) + "?", "answer": " ".join(words)}
            kind = "synthetic"
        else:
            return super().complete(messages, **kwargs)
        self.kinds.append(kind)
        return LLMResult(json.dumps(out), self.model, 30, 10)


CHUNKS = [("c1", "alpha text"), ("c2", "beta text"), ("c3", "gamma text")]


def test_context_precision_judge(client: TestClient) -> None:  # client: initialises the DB
    llm = JudgeLLM()
    out = judge.context_precision(llm, "q?", CHUNKS, 42)
    assert out["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert [v["relevant"] for v in out["verdicts"]] == [True, False, True]
    assert out["cached"] is False
    again = judge.context_precision(llm, "q?", CHUNKS, 42)
    assert again["cached"] is True and llm.kinds == ["precision"]
    # with a reference answer it is a different judgement (and cache entry)
    judge.context_precision(llm, "q?", CHUNKS, 42, ground_truth="alpha")
    assert llm.kinds == ["precision", "precision"]
    assert judge.context_precision(llm, "q?", [], 42)["value"] is None


def test_precision_judge_that_skips_passages_is_unavailable(client: TestClient) -> None:
    class Lazy(JudgeLLM):
        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201
            return LLMResult('{"passages": [{"index": 1, "relevant": true}]}', "m", 1, 1)

    out = judge.context_precision(Lazy(), "q?", CHUNKS, 42)
    assert out["value"] is None and "skipped some passages" in out["note"]


def test_recall_and_correctness_judges(client: TestClient) -> None:
    llm = JudgeLLM()
    assert judge.context_recall(llm, "truth", CHUNKS, 1)["value"] == 0.5
    assert judge.context_recall(llm, "truth", [], 1)["value"] == 0.0
    out = judge.answer_correctness(llm, FakeEmbedder(), "q?", "alpha beta", "alpha beta", 1)
    # F1 = 1 / (1 + 0.5 * 1) = 0.6667; identical texts -> similarity 1.0
    assert out["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert (out["tp"], out["fp"], out["fn"]) == (["a"], [], ["b"])  # extra "x" is not an fp
    assert out["value"] == pytest.approx(0.75 * 2 / 3 + 0.25, abs=1e-3)


def test_judge_errors_become_notes(client: TestClient) -> None:
    class Down(JudgeLLM):
        def complete(self, messages, **kwargs):  # noqa: ANN001, ANN201
            raise LLMError("provider down")

    for out in (
        judge.context_precision(Down(), "q?", CHUNKS, 1),
        judge.context_recall(Down(), "t", CHUNKS, 1),
        judge.answer_correctness(Down(), FakeEmbedder(), "q?", "a", "t", 1),
        judge.faithfulness(Down(), "Some claim.", "ctx", 1),
    ):
        assert out["value"] is None and "provider down" in out["note"]


def test_faithfulness_reuses_q9_verdicts_only_for_the_same_text() -> None:
    rows = [{"text": "a", "supported": True}, {"text": "b", "supported": False}]
    llm = JudgeLLM()
    out = judge.faithfulness(llm, "Final.", "ctx", 1, rows, "Final.")
    assert out["value"] == 0.5 and out["source"] == "q9" and llm.kinds == []
    out = judge.faithfulness(llm, "Changed.", "ctx", 1, rows, "Final.")
    assert out["source"] == "judge" and llm.kinds == ["groundedness"]
    assert judge.faithfulness(llm, NOT_FOUND, "ctx", 1)["value"] is None


# ---------------------------------------------------------------- Q10 via the API
@pytest.fixture
def llm() -> JudgeLLM:
    return JudgeLLM()


@pytest.fixture
def eval_client(app: FastAPI, llm: JudgeLLM) -> Iterator[TestClient]:
    app.state.embedder = FakeEmbedder()
    app.state.reranker = FakeReranker()
    app.state.llm_factory = lambda: llm
    with TestClient(app) as c:
        cfg = c.get("/api/config").json()
        # the hashed test embedder makes every query look off-topic
        cfg["guardrails"]["input"]["off_topic"]["enabled"] = False
        assert c.put("/api/config", json=cfg).status_code == 200
        yield c


def _ingest(client: TestClient) -> None:
    ingest_ok(client, [("notes.txt", make_text(seed=3, paragraphs=40).encode())])


def _query(client: TestClient, query: str = QUERY) -> tuple[dict, list[dict], dict]:
    body = {"query": query, "params": {"similarity_threshold": 0.0, "top_n": 2}}
    run_id = client.post("/api/query", json=body).json()["run_id"]
    events = parse_sse(client.get(f"/api/query/{run_id}/events").text)
    return final_stages(events), events, events[-1]["data"]


def test_q10_scores_after_the_answer_is_published(eval_client: TestClient, llm: JudgeLLM) -> None:
    _ingest(eval_client)
    stages, events, done = _query(eval_client)
    order = [(e["event"], e["data"].get("stage_id"), e["data"].get("status")) for e in events]
    answer_at = next(i for i, o in enumerate(order) if o[0] == "answer")
    assert order.index(("stage", "Q9_output_guardrail", "success")) < answer_at
    assert answer_at < order.index(("stage", "Q10_eval", "running"))
    assert events[answer_at]["data"]["answer"] == done["answer"]

    q10 = stages["Q10_eval"]
    assert q10["status"] == "success", q10["summary"]
    m = q10["data"]["metrics"]
    assert m["faithfulness"]["value"] == 1.0
    assert m["faithfulness"]["source"] == "q9"  # Q9's judgement reused: no second call
    assert llm.kinds.count("groundedness") == 1
    assert m["context_precision"]["value"] == 0.5  # 2 chunks, the judge marks odd ones
    assert 0.0 <= m["answer_relevancy"]["value"] <= 1.0

    tokens = q10["data"]["tokens"]
    assert tokens["by_stage"]["Q8_generate"]["calls"] == 1
    assert tokens["by_stage"]["Q9_output_guardrail"]["calls"] == 1
    assert tokens["by_stage"]["Q10_eval"]["calls"] == 1
    latency = q10["data"]["latency_ms"]
    assert set(latency["stages"]) >= {"Q1_validate", "Q8_generate", "Q9_output_guardrail"}
    assert latency["answer_ready"] >= latency["stages"]["Q8_generate"]
    assert done["eval"] == q10["data"]

    # The same query again: the Q5 entities and the precision judgement come from the cache.
    stages, _, _ = _query(eval_client)
    by_stage = stages["Q10_eval"]["data"]["tokens"]["by_stage"]
    assert by_stage["Q10_eval"] == {
        "calls": 0,
        "cache_hits": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    assert by_stage["Q5_graph_retrieve"]["cache_hits"] == 1


def test_q10_can_be_turned_off(eval_client: TestClient, llm: JudgeLLM) -> None:
    cfg = eval_client.get("/api/config").json()
    assert cfg["evaluation"]["online"]["enabled"] is True
    cfg["evaluation"]["online"]["enabled"] = False
    assert eval_client.put("/api/config", json=cfg).status_code == 200
    _ingest(eval_client)
    stages, _, done = _query(eval_client)
    assert stages["Q10_eval"]["data"] == {"skipped": True}
    assert done["eval"] is None
    assert "precision" not in llm.kinds


def test_q10_without_context(eval_client: TestClient, llm: JudgeLLM) -> None:
    _ingest(eval_client)
    body = {"query": QUERY, "params": {"similarity_threshold": 1.0, "graph_hops": 0}}
    run_id = eval_client.post("/api/query", json=body).json()["run_id"]
    events = parse_sse(eval_client.get(f"/api/query/{run_id}/events").text)
    m = final_stages(events)["Q10_eval"]["data"]["metrics"]
    assert all(v["value"] is None for v in m.values()), m
    assert llm.kinds == []


# ---------------------------------------------------------------- golden sets
def _chunk_ids(client: TestClient, n: int = 3) -> list[str]:
    return [c["id"] for c in client.get("/api/chunks", params={"limit": n}).json()["items"]]


def _upload(client: TestClient, lines: list, name: str = "gold"):  # noqa: ANN202
    body = "\n".join(json.dumps(x) if not isinstance(x, str) else x for x in lines)
    return client.post(
        "/api/eval/datasets",
        data={"name": name},
        files={"file": ("gold.jsonl", body.encode(), "application/jsonl")},
    )


def test_upload_golden_set_resolves_chunk_ids_to_spans(eval_client: TestClient) -> None:
    _ingest(eval_client)
    ids = _chunk_ids(eval_client)
    resp = _upload(
        eval_client,
        [
            {
                "question": QUERY,
                "ground_truth": "Cosine compares vectors.",
                "relevant_chunk_ids": ids[:2],
            },
            {"question": "What is a vertex?", "ground_truth": "A graph node."},
        ],
    )
    assert resp.status_code == 201, resp.text
    ds = resp.json()
    assert ds["status"] == "ready" and ds["question_count"] == 2
    detail = eval_client.get(f"/api/eval/datasets/{ds['id']}").json()
    spans = detail["questions"][0]["relevant_spans"]
    assert [s["chunk_id"] for s in spans] == ids[:2]
    assert all({"sha256", "start", "end", "filename"} <= set(s) for s in spans)
    assert detail["questions"][1]["relevant_spans"] == []


def test_upload_reports_errors_per_line(eval_client: TestClient) -> None:
    resp = _upload(
        eval_client,
        [
            {"question": "ok question", "ground_truth": "fine"},
            "{not json",
            {"question": "x", "ground_truth": "too short a question"},
            {"question": "unknown ids?", "ground_truth": "t", "relevant_chunk_ids": ["zz-v1-0"]},
            {"question": "extra key?", "ground_truth": "t", "bonus": 1},
        ],
    )
    assert resp.status_code == 422
    fields = {d["field"] for d in resp.json()["details"]}
    assert "line 2" in fields
    assert "line 3.question" in fields
    assert "line 4.relevant_chunk_ids" in fields
    assert "line 5.bonus" in fields


def test_synthetic_set_is_seeded(eval_client: TestClient, llm: JudgeLLM) -> None:
    _ingest(eval_client)

    def make(seed: int) -> list[dict]:
        resp = eval_client.post(
            "/api/eval/datasets/synthetic", json={"name": f"s{seed}", "count": 3, "seed": seed}
        )
        assert resp.status_code == 202, resp.text
        ds = eval_client.get(f"/api/eval/datasets/{resp.json()['id']}").json()
        assert ds["status"] == "ready" and ds["question_count"] == 3, ds
        return ds["questions"]

    a, b, c = make(7), make(7), make(8)
    key = [q["relevant_spans"][0]["chunk_id"] for q in a]
    assert key == [q["relevant_spans"][0]["chunk_id"] for q in b]
    assert key != [q["relevant_spans"][0]["chunk_id"] for q in c]
    assert [q["question"] for q in a] == [q["question"] for q in b]


def test_synthetic_without_chunks_errors(eval_client: TestClient) -> None:
    resp = eval_client.post("/api/eval/datasets/synthetic", json={"name": "empty", "count": 2})
    ds = eval_client.get(f"/api/eval/datasets/{resp.json()['id']}").json()
    assert ds["status"] == "error" and "no unique chunks" in ds["error"]


# ---------------------------------------------------------------- eval runs
def _dataset(client: TestClient) -> tuple[str, list[str]]:
    ids = _chunk_ids(client, 40)
    resp = _upload(
        client,
        [
            {
                "question": QUERY,
                "ground_truth": "Cosine similarity compares embedding vectors.",
                "relevant_chunk_ids": ids,
            },
            {
                "question": "how do graph edges link vertices",
                "ground_truth": "Edges join vertices.",
            },
        ],
    )
    return resp.json()["id"], ids


@pytest.mark.parametrize(
    ("grid", "field"),
    [
        ({"chunk_overlap": [16, 32]}, "param_grid.chunk_overlap"),
        ({"chunk_size": [100]}, "param_grid.cell 1.chunk_size"),
        ({"bogus": [1]}, "param_grid.bogus"),
        ({"top_k": [5, 5]}, "param_grid.top_k"),
        ({"top_k": [2], "top_n": [4]}, "param_grid.cell 1.top_n"),
        ({"top_k": [5, 6, 7, 8], "top_n": [1, 2, 3, 4]}, "param_grid"),
        ({"top_k": [5.5]}, "param_grid.cell 1.top_k"),
    ],
)
def test_grid_validation(eval_client: TestClient, grid: dict, field: str) -> None:
    _ingest(eval_client)
    ds_id, _ = _dataset(eval_client)
    resp = eval_client.post("/api/eval/runs", json={"dataset_id": ds_id, "param_grid": grid})
    assert resp.status_code == 422, resp.text
    assert field in {d["field"] for d in resp.json()["details"]}


def test_eval_run_end_to_end(eval_client: TestClient, llm: JudgeLLM) -> None:
    _ingest(eval_client)
    ds_id, ids = _dataset(eval_client)
    resp = eval_client.post(
        "/api/eval/runs",
        json={"dataset_id": ds_id, "param_grid": {"top_n": [1, 2], "similarity_threshold": [0.0]}},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["eval_run_id"]

    run = eval_client.get(f"/api/eval/runs/{run_id}").json()
    assert run["status"] == "success", run["error"]
    assert run["total"] == run["done"] == 4
    assert [c["top_n"] for c in run["cells"]] == [1, 2]
    assert [s["params"]["top_n"] for s in run["summary"]] == [1, 2]
    metrics = run["summary"][1]["metrics"]
    assert set(metrics) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "hit_rate",
        "mrr",
    }
    assert metrics["hit_rate"] == {"mean": 1.0, "n": 1}  # only question 1 has relevant chunks
    assert metrics["context_recall"]["mean"] == 0.5
    assert metrics["faithfulness"]["mean"] == 1.0

    rows = run["results"]
    assert [r["question"] for r in rows] == [QUERY, "how do graph edges link vertices"]
    assert rows[1]["metrics"]["hit_rate"] is None
    detail = eval_client.get(f"/api/eval/results/{rows[0]['id']}").json()
    assert [r["rank"] for r in detail["retrieved"]] == [1]
    assert detail["retrieved"][0]["relevant"] is True
    assert detail["details"]["context_precision"]["verdicts"][0]["relevant"] is True
    assert detail["details"]["answer_correctness"]["fn"] == ["b"]
    assert detail["tokens"]["calls"] >= 4

    cell2 = eval_client.get(f"/api/eval/runs/{run_id}", params={"cell": 1}).json()["results"]
    assert len(cell2) == 2
    # eval queries stay out of the live run history
    assert eval_client.get("/api/runs", params={"kind": "query"}).json()["total"] == 0
    listed = eval_client.get("/api/eval/runs").json()
    assert listed[0]["id"] == run_id and listed[0]["dataset_name"] == "gold"


def test_eval_run_requires_a_corpus(eval_client: TestClient) -> None:
    ds_id = _upload(eval_client, [{"question": "q one?", "ground_truth": "t"}]).json()["id"]
    resp = eval_client.post("/api/eval/runs", json={"dataset_id": ds_id})
    assert resp.status_code == 409


def test_eval_limits(eval_client: TestClient) -> None:
    out = eval_client.get("/api/eval/limits").json()
    assert {"chunk_size", "top_k"} <= set(out["grid_params"])
    assert "chunk_overlap" not in out["grid_params"]
    assert out["llm_calls_per_question"] == 2 and out["llm_calls_per_cell"] == 8
    assert out["max_cells"] == 12


def test_cancelled_run_keeps_a_partial_summary(
    eval_client: TestClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.eval import runner

    _ingest(eval_client)
    ds_id, _ = _dataset(eval_client)
    evaluate = runner.EvalJob._evaluate

    def cancel_after_first(self, cell, q, params, target):  # noqa: ANN001, ANN202
        result = evaluate(self, cell, q, params, target)
        runner.request_cancel(self.id)
        return result

    monkeypatch.setattr(runner.EvalJob, "_evaluate", cancel_after_first)
    resp = eval_client.post(
        "/api/eval/runs",
        json={"dataset_id": ds_id, "param_grid": {"similarity_threshold": [0.0]}},
    )
    run = eval_client.get(f"/api/eval/runs/{resp.json()['eval_run_id']}").json()
    assert run["status"] == "cancelled" and run["done"] == 1 and run["total"] == 2
    assert len(run["summary"]) == 1
    assert run["summary"][0]["partial"] is True and run["summary"][0]["questions"] == 1
    assert len(run["results"]) == 1


# ---------------------------------------------------------------- step 7b: chunk_size grids
def test_use_engine_scopes_the_override(client: TestClient, tmp_path) -> None:  # noqa: ANN001
    from app.db.session import create_db, get_engine, use_engine

    live = get_engine()
    other = create_db(tmp_path / "other" / "x.db")
    with use_engine(other):
        assert get_engine() is other
        with use_engine(None):
            assert get_engine() is live
        assert get_engine() is other
    assert get_engine() is live


def test_ingest_keeps_the_original_and_delete_removes_it(
    eval_client: TestClient, app: FastAPI
) -> None:
    _ingest(eval_client)
    doc = eval_client.get("/api/documents").json()[0]
    raw = app.state.settings.uploads_dir / doc["sha256"]
    assert raw.read_bytes() == make_text(seed=3, paragraphs=40).encode()
    assert eval_client.delete(f"/api/documents/{doc['id']}").status_code == 200
    assert not raw.exists()


def test_ingest_and_delete_are_blocked_during_an_eval(eval_client: TestClient) -> None:
    from app.eval import runner

    _ingest(eval_client)
    doc_id = eval_client.get("/api/documents").json()[0]["id"]
    assert runner.try_start("held")
    try:
        files = {"files": ("b.txt", b"some other text for the corpus", "text/plain")}
        assert eval_client.post("/api/ingest", files=files).status_code == 409
        assert eval_client.delete(f"/api/documents/{doc_id}").status_code == 409
    finally:
        runner.release("held")


def _kg_calls(llm: JudgeLLM) -> int:
    return sum("knowledge graph from one passage" in m[0]["content"] for m in llm.complete_calls)


def _counts(client: TestClient) -> tuple:
    """Live state an index build must not touch: documents, chunks (with their cluster and
    duplicate links), clusters, Chroma metadata and the graph."""
    docs = client.get("/api/documents").json()
    chunks = client.get("/api/chunks", params={"limit": 200}).json()["items"]
    graph = client.get("/api/graph").json()["stats"]
    meta = client.app.state.vector_store._existing().get(include=["metadatas"])
    return (
        [(d["id"], d["version"], d["chunk_count"]) for d in docs],
        [(c["id"], c["cluster_id"], c["is_duplicate_of"]) for c in chunks],
        client.get("/api/clusters").json(),
        sorted(zip(meta["ids"], [sorted(m.items()) for m in meta["metadatas"]], strict=True)),
        (graph["vertices"], graph["edges"]),
    )


def test_chunk_size_grid_uses_throwaway_indexes(eval_client: TestClient, llm: JudgeLLM) -> None:
    _ingest(eval_client)
    ds_id, _ = _dataset(eval_client)
    before = _counts(eval_client)
    body = {
        "dataset_id": ds_id,
        "param_grid": {"chunk_size": [128, 256], "similarity_threshold": [0.0]},
    }
    est = eval_client.post("/api/eval/estimate", json=body).json()
    assert [b["chunk_size"] for b in est["index_builds"]] == [128, 256]
    assert not any(b["cached"] for b in est["index_builds"])
    assert est["index_builds"][0]["chunks"] > est["index_builds"][1]["chunks"] > 0
    assert est["total_calls_max"] > est["query_calls_max"] and est["missing_sources"] == []

    kg_before = _kg_calls(llm)
    run_id = eval_client.post("/api/eval/runs", json=body).json()["eval_run_id"]
    run = eval_client.get(f"/api/eval/runs/{run_id}").json()
    assert run["status"] == "success", run["error"]
    assert [c["chunk_size"] for c in run["cells"]] == [128, 256]
    assert [s["params"]["chunk_size"] for s in run["summary"]] == [128, 256]
    # the golden chunks were ids of the live 512-token index; spans still find them
    assert run["summary"][0]["metrics"]["hit_rate"]["mean"] == 1.0
    detail = eval_client.get(f"/api/eval/results/{run['results'][0]['id']}").json()
    assert detail["retrieved"] and not detail["retrieved"][0]["chunk_id"].startswith(
        detail["relevant_spans"][0]["chunk_id"][:8]
    )  # a throwaway index's chunk, not a live one
    built = _kg_calls(llm) - kg_before
    assert built > 0

    idx = eval_client.get("/api/eval/indexes").json()
    assert sorted(i["chunk_size"] for i in idx) == [128, 256]
    by_size = {b["chunk_size"]: b["chunks"] for b in est["index_builds"]}
    assert {i["chunk_size"]: i["chunk_count"] for i in idx} == by_size  # the estimate is exact
    assert all(i["status"] == "ready" and not i["stale"] for i in idx)
    assert all(i["chunk_count"] > 0 and i["build_tokens"]["calls"] > 0 for i in idx)
    assert _counts(eval_client) == before  # the live corpus, index and graph are untouched
    assert eval_client.get("/api/runs", params={"kind": "ingest"}).json()["total"] == 1

    # A second run reuses both indexes: no new KG extraction.
    est = eval_client.post("/api/eval/estimate", json=body).json()
    assert all(b["cached"] and b["max_calls"] == 0 for b in est["index_builds"])
    kg_before = _kg_calls(llm)
    run_id = eval_client.post("/api/eval/runs", json=body).json()["eval_run_id"]
    assert eval_client.get(f"/api/eval/runs/{run_id}").json()["status"] == "success"
    assert _kg_calls(llm) == kg_before
    assert len(eval_client.get("/api/eval/indexes").json()) == 2


def test_new_corpus_marks_indexes_stale_and_delete_removes_them(
    eval_client: TestClient, app: FastAPI
) -> None:
    _ingest(eval_client)
    ds_id, _ = _dataset(eval_client)
    body = {"dataset_id": ds_id, "param_grid": {"chunk_size": [256]}}
    run_id = eval_client.post("/api/eval/runs", json=body).json()["eval_run_id"]
    run = eval_client.get(f"/api/eval/runs/{run_id}").json()
    assert run["status"] == "success", run["error"]
    ingest_ok(eval_client, [("more.txt", make_text(seed=9, paragraphs=10).encode())])
    (idx,) = eval_client.get("/api/eval/indexes").json()
    assert idx["stale"] is True
    est = eval_client.post("/api/eval/estimate", json=body).json()
    assert est["index_builds"][0]["cached"] is False  # the new corpus needs a new index

    root = app.state.settings.eval_indexes_dir
    assert any(root.iterdir())
    assert eval_client.delete(f"/api/eval/indexes/{idx['id']}").status_code == 200
    assert eval_client.get("/api/eval/indexes").json() == []
    assert not any(root.iterdir())
    assert eval_client.delete(f"/api/eval/indexes/{idx['id']}").status_code == 404


def test_chunk_size_needs_the_original_uploads(eval_client: TestClient, app: FastAPI) -> None:
    _ingest(eval_client)
    ds_id, _ = _dataset(eval_client)
    for f in app.state.settings.uploads_dir.iterdir():  # as if ingested before 7b
        f.unlink()
    body = {"dataset_id": ds_id, "param_grid": {"chunk_size": [256]}}
    assert eval_client.post("/api/eval/estimate", json=body).json()["missing_sources"] == [
        "notes.txt"
    ]
    resp = eval_client.post("/api/eval/runs", json=body)
    assert resp.status_code == 409 and "notes.txt" in resp.text
    # re-uploading the same file with the same settings only stores the original
    ingest_ok(eval_client, [("notes.txt", make_text(seed=3, paragraphs=40).encode())])
    assert eval_client.post("/api/eval/estimate", json=body).json()["missing_sources"] == []


def test_relevance_across_chunk_sizes_discriminates(eval_client: TestClient) -> None:
    """One middle 512-token chunk is golden: smaller chunks inside it match, others do not."""
    _ingest(eval_client)
    live = eval_client.get("/api/chunks", params={"limit": 50}).json()["items"]
    assert len(live) >= 3
    golden = live[len(live) // 2]
    resp = _upload(
        eval_client,
        [
            {
                "question": QUERY,
                "ground_truth": "Cosine compares vectors.",
                "relevant_chunk_ids": [golden["id"]],
            }
        ],
    )
    grid = {"chunk_size": [128], "similarity_threshold": [0.0], "top_k": [40], "top_n": [20]}
    run_id = eval_client.post(
        "/api/eval/runs", json={"dataset_id": resp.json()["id"], "param_grid": grid}
    ).json()["eval_run_id"]
    run = eval_client.get(f"/api/eval/runs/{run_id}").json()
    assert run["status"] == "success", run["error"]
    retrieved = eval_client.get(f"/api/eval/results/{run['results'][0]['id']}").json()["retrieved"]
    flags = [r["relevant"] for r in retrieved]
    assert True in flags and False in flags, flags
    for r in retrieved:  # >= 50% overlap: some end lies inside; fully inside: always relevant
        head, tail = r["text"][:40] in golden["text"], r["text"][-40:] in golden["text"]
        if r["relevant"]:
            assert head or tail, r["chunk_id"]
        if r["text"] in golden["text"]:
            assert r["relevant"], r["chunk_id"]
