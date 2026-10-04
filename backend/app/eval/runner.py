"""Offline evaluation: run every golden question through the full query pipeline (Q1-Q10) for
each cell of a parameter grid, then score it.

Questions run headless: ``CaptureBus`` stands in for the SSE bus, so eval queries emit no SSE,
persist no stage events and never appear in the live run history. Q10 supplies faithfulness and
answer relevancy; this module adds the ground-truth metrics (context precision/recall, answer
correctness via LLM judges) and the retrieval metrics (hit rate@top_n, MRR via span overlap).
One eval run executes at a time; it can be cancelled between questions.

When the grid includes chunk_size, every cell queries a throwaway index built at that chunk
size (see ``app.eval.indexes``); the query and the span lookup run inside ``use_engine`` so they
read that index's database, while results and judge caching stay in the live database.
"""

import itertools
import logging
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, col, select

from app.core.logging import run_id_var
from app.db.models import (
    Chunk,
    Document,
    EvalDataset,
    EvalIndex,
    EvalQuestion,
    EvalResult,
    EvalRun,
)
from app.db.session import get_engine, use_engine
from app.eval import indexes, judge
from app.eval.capture import CaptureBus
from app.eval.metrics import hit_rate, reciprocal_rank, relevance_flags, summarise
from app.llm.usage import MeteredLLM, UsageMeter, active_meter
from app.pipeline.query.runner import QueryDeps, QueryJob
from app.schemas.config import GuardrailSettings
from app.schemas.params import IngestParams, QueryParams
from app.schemas.query import QueryRequest

logger = logging.getLogger(__name__)

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "hit_rate",
    "mrr",
)
MAX_CELLS = 12
MAX_QUESTIONS = 200
# Worst-case LLM calls. Per question: the Q2 classifier and Q5 entity extraction, cached by
# question text so they are paid once, not per cell. Per question per cell: Q8 answer, Q9 judge,
# regeneration and re-judge, Q10 faithfulness re-judge (when Q9 changed the text), and the
# precision, recall and correctness judges. Cache hits make real runs cheaper.
LLM_CALLS_PER_QUESTION = 2
LLM_CALLS_PER_CELL = 8
GRID_PARAMS = ("chunk_size", *QueryParams.model_fields)
NOT_IN_GRID = ("chunk_overlap", "dedup_threshold", "build_graph")
INGEST_KEYS = tuple(IngestParams.model_fields)  # stored in a cell when chunk_size is in the grid

_cancel: dict[str, threading.Event] = {}
_lock = threading.Lock()


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid values, in the grid's key order ({} -> one default cell)."""
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def try_start(run_id: str) -> bool:
    """Reserve the single eval slot for ``run_id``; False if another run holds it."""
    with _lock:
        if _cancel:
            return False
        _cancel[run_id] = threading.Event()
        return True


def is_running() -> bool:
    with _lock:
        return bool(_cancel)


def release(run_id: str) -> None:
    with _lock:
        _cancel.pop(run_id, None)


def request_cancel(run_id: str) -> bool:
    with _lock:
        event = _cancel.get(run_id)
    if event is not None:
        event.set()
    return event is not None


def recover_interrupted() -> None:
    """Eval runs / synthetic generations left in progress by a server restart can never finish:
    mark them as errors."""
    note = "interrupted (server restarted)"
    with Session(get_engine()) as s:
        for run in s.exec(select(EvalRun).where(EvalRun.status == "running")).all():
            run.status, run.error, run.finished_at = "error", note, _now()
            s.add(run)
        for ds in s.exec(select(EvalDataset).where(EvalDataset.status == "generating")).all():
            ds.status, ds.error = "error", note
            s.add(ds)
        for idx in s.exec(select(EvalIndex).where(EvalIndex.status == "building")).all():
            idx.status, idx.error = "error", note  # rebuilt in a fresh directory when next used
            s.add(idx)
        s.commit()


def _now() -> datetime:
    return datetime.now(UTC)


def chunk_spans(ids: list[str]) -> dict[str, dict[str, Any]]:
    """chunk id -> {sha256, start, end, text, filename, page} for the chunks that still exist."""
    if not ids:
        return {}
    with Session(get_engine()) as s:
        rows = s.exec(
            select(Chunk, Document)
            .join(Document, col(Document.id) == col(Chunk.document_id))
            .where(col(Chunk.id).in_(ids))
        ).all()
    return {
        c.id: {
            "sha256": d.sha256,
            "start": c.start_offset,
            "end": c.end_offset,
            "text": c.text,
            "filename": d.filename,
            "page": c.page,
        }
        for c, d in rows
    }


class EvalJob:
    def __init__(
        self,
        eval_run_id: str,
        deps: QueryDeps,
        guardrails: GuardrailSettings,
        collection: str,
        ingest_seed: int,
    ):
        self.id = eval_run_id
        self.deps = deps
        self.guardrails = guardrails
        self.collection = collection
        self.ingest_seed = ingest_seed
        # (cell, cell params, scored rows) of the cell in progress; summarised as partial if the
        # run stops (cancel or error) before the cell completes.
        self._open: tuple[int, dict[str, Any], list[dict[str, Any]]] | None = None

    def run(self) -> None:
        run_id_var.set(self.id)
        status, error = "error", None
        try:
            status = self._run()
        except Exception as exc:
            logger.exception("eval run failed")
            error = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            if self._open and self._open[2]:
                self._save_summary(*self._open, partial=True)
            with Session(get_engine()) as s:
                run = s.get(EvalRun, self.id)
                if run is not None:
                    run.status, run.error, run.finished_at = status, error, _now()
                    s.add(run)
                    s.commit()
            release(self.id)

    def _run(self) -> str:
        with Session(get_engine()) as s:
            run = s.get(EvalRun, self.id)
            assert run is not None
            cells = list(run.cells)
            questions = s.exec(
                select(EvalQuestion)
                .where(EvalQuestion.dataset_id == run.dataset_id)
                .order_by(col(EvalQuestion.idx))
            ).all()
        cancel = _cancel[self.id]
        for cell, values in enumerate(cells):
            params = QueryParams.model_validate(
                {k: v for k, v in values.items() if k not in INGEST_KEYS}
            )
            target = self._target(values)
            rows: list[dict[str, Any]] = []
            self._open = (cell, values, rows)
            for q in questions:
                if cancel.is_set():
                    return "cancelled"
                result = self._evaluate(cell, q, params, target)
                rows.append(result.model_dump())  # plain copy: the row expires on commit
                with Session(get_engine()) as s:
                    s.add(result)
                    run = s.get(EvalRun, self.id)
                    assert run is not None
                    run.done += 1
                    s.add(run)
                    s.commit()
            self._open = None
            self._save_summary(cell, values, rows)
        return "success"

    def _target(self, values: dict[str, Any]) -> tuple[QueryDeps, str, Any]:
        """(deps, collection, engine or None) the cell queries: the live index, or the
        throwaway index for its chunk size (built now unless cached)."""
        if "chunk_size" not in values:
            return self.deps, self.collection, None
        ingest = IngestParams.model_validate({k: values[k] for k in INGEST_KEYS})
        shadow = indexes.ensure(
            ingest, self.ingest_seed, self.deps, self.deps.settings, get_engine()
        )
        deps = replace(self.deps, vector_store=shadow.vector_store, graph_store=shadow.graph_store)
        return deps, shadow.collection, shadow.engine

    def _save_summary(
        self, cell: int, params: dict[str, Any], rows: list[dict[str, Any]], partial: bool = False
    ) -> None:
        tokens = sum(
            r["tokens"].get("prompt_tokens", 0) + r["tokens"].get("completion_tokens", 0)
            for r in rows
        )
        statuses = sorted({r["status"] for r in rows})
        entry = {
            "cell": cell,
            "params": params,
            "metrics": summarise([r["metrics"] for r in rows]),
            "questions": len(rows),
            "statuses": {st: sum(r["status"] == st for r in rows) for st in statuses},
            "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows)) if rows else 0,
            "total_tokens": tokens,
            "partial": partial,
        }
        with Session(get_engine()) as s:
            run = s.get(EvalRun, self.id)
            assert run is not None
            run.summary = [*run.summary, entry]
            s.add(run)
            s.commit()

    def _evaluate(
        self, cell: int, q: EvalQuestion, params: QueryParams, target: tuple[QueryDeps, str, Any]
    ) -> EvalResult:
        base, collection, engine = target
        scope = use_engine(engine) if engine is not None else nullcontext()
        bus = CaptureBus()
        deps = replace(base, bus=bus)
        request = QueryRequest(query=q.question, collection=collection)
        job = QueryJob(
            str(uuid.uuid4()),
            request,
            params,
            deps,
            self.guardrails,
            online_eval=True,
            precision_judge=False,
        )
        t0 = time.perf_counter()
        with scope:
            job.run()
        latency = round((time.perf_counter() - t0) * 1000)
        done, stages = bus.done, bus.stages
        status = done.get("status", "error")
        result = EvalResult(
            eval_run_id=self.id,
            cell=cell,
            question_id=q.id,
            status=status,
            answer=str(done.get("answer", "")),
            error=done.get("error"),
            latency_ms=latency,
        )
        usage = job.meter.summary()["total"]
        if status != "success":
            result.metrics = dict.fromkeys(METRICS)
            result.tokens = usage
            return result

        # Retrieved = the top_n context sent to the LLM, in rank order; each item also carries
        # the near-duplicates Q4 collapsed onto it (they hold the same text).
        context_ids: list[str] = (
            stages.get("Q8_generate", {}).get("data", {}).get("context_chunk_ids", [])
        )
        q4 = stages.get("Q4_vector_retrieve", {}).get("data", {}).get("candidates", [])
        collapsed = {c["chunk_id"]: c.get("collapsed_from", []) for c in q4}
        with use_engine(engine) if engine is not None else nullcontext():
            spans = chunk_spans(
                context_ids + [d for cid in context_ids for d in collapsed.get(cid, [])]
            )
        flags = relevance_flags(
            [
                [spans[i] for i in [cid, *collapsed.get(cid, [])] if i in spans]
                for cid in context_ids
            ],
            q.relevant_spans,
        )
        q10 = stages.get("Q10_eval", {}).get("data", {}).get("metrics", {})
        chunks = [(cid, spans[cid]["text"]) for cid in context_ids if cid in spans]

        meter = UsageMeter(lambda: "eval_judges")
        active_meter.set(meter)
        seed = params.seed
        try:
            llm = MeteredLLM(self.deps.llm_factory(), meter)
        except Exception as exc:  # e.g. no API key: the no-context path never needed the LLM
            na = {"value": None, "note": f"judge unavailable ({type(exc).__name__}: {exc})"[:200]}
            precision, recall, correctness = na, na, na
        else:
            precision = judge.context_precision(llm, q.question, chunks, seed, q.ground_truth)
            recall = judge.context_recall(llm, q.ground_truth, chunks, seed)
            correctness = judge.answer_correctness(
                llm, self.deps.embedder, q.question, result.answer, q.ground_truth, seed
            )
        has_relevant = bool(q.relevant_spans)
        result.metrics = {
            "faithfulness": q10.get("faithfulness", {}).get("value"),
            "answer_relevancy": q10.get("answer_relevancy", {}).get("value"),
            "context_precision": precision["value"],
            "context_recall": recall["value"],
            "answer_correctness": correctness["value"],
            "hit_rate": hit_rate(flags) if has_relevant else None,
            "mrr": reciprocal_rank(flags) if has_relevant else None,
        }
        result.retrieved = [
            {
                "rank": rank,
                "chunk_id": cid,
                "filename": spans.get(cid, {}).get("filename"),
                "page": spans.get(cid, {}).get("page"),
                "text": spans.get(cid, {}).get("text", ""),
                "relevant": flag if has_relevant else None,
            }
            for rank, (cid, flag) in enumerate(zip(context_ids, flags, strict=True), start=1)
        ]
        result.details = {
            "faithfulness": q10.get("faithfulness", {}),
            "answer_relevancy": q10.get("answer_relevancy", {}),
            "context_precision": precision,
            "context_recall": recall,
            "answer_correctness": correctness,
            "guardrails": done.get("guardrails", {}),
        }
        judged = meter.summary()["total"]
        result.tokens = {k: usage[k] + judged[k] for k in usage}
        return result
