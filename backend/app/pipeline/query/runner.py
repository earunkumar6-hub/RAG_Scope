"""Query run orchestration: Q1 validate -> Q9 output guardrail, one StageEvent per stage.

Q2 screens the (already PII-masked) query and may block the run. Q5 expands the knowledge graph
around the query's entities; Q6 fuses its evidence chunks with the vector candidates (weighted
RRF) before Q7 re-ranks them. Q8 streams the answer as ``token`` messages; Q9 then checks and,
if needed, corrects it. The final answer is then published as an ``answer`` message, so the
client can show it while Q10 scores it (faithfulness, answer relevancy, context precision,
per-stage latency and LLM token usage).
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.events import EventBus
from app.core.logging import run_id_var
from app.db.models import Run
from app.db.session import get_engine
from app.eval import judge
from app.eval.metrics import answer_relevancy
from app.graph.base import GraphStore
from app.guardrails.input_guards import (
    BLOCKED_ANSWER,
    CentroidCache,
    GuardCheck,
    InputReport,
    run_input_guards,
)
from app.guardrails.output_guards import NOT_FOUND, OutputResult, run_output_guards
from app.guardrails.pii import get_pii_engine
from app.llm.base import LLMProvider
from app.llm.usage import MeteredLLM, UsageMeter, active_meter
from app.pipeline.ingestion.embedder import Embedder
from app.pipeline.ingestion.tokenizer import get_tokenizer
from app.pipeline.ingestion.vector_store import Neighbour, VectorStore
from app.pipeline.query import graph_retriever
from app.pipeline.query.fusion import RRF_K, weighted_rrf
from app.pipeline.query.graph_retriever import GraphResult
from app.pipeline.query.reranker import MAX_PAIR_TOKENS, Reranker
from app.pipeline.stages import StageRecorder
from app.schemas.config import GuardrailSettings
from app.schemas.events import QueryStageId
from app.schemas.params import QueryParams
from app.schemas.query import QueryFilters, QueryRequest

logger = logging.getLogger(__name__)

QUERY_STAGES: tuple[QueryStageId, ...] = (
    "Q1_validate",
    "Q2_input_guardrail",
    "Q3_query_embed",
    "Q4_vector_retrieve",
    "Q5_graph_retrieve",
    "Q6_fusion",
    "Q7_rerank",
    "Q8_generate",
    "Q9_output_guardrail",
    "Q10_eval",
)
NO_CONTEXT_ANSWER = NOT_FOUND
SYSTEM_PROMPT = """You answer questions about a document collection.
Rules:
1. Answer ONLY from the context below. Do not use outside knowledge.
2. After each claim, cite the supporting chunk id in square brackets, e.g. [0a1b2c3d-v1-4]. \
Cite only ids that appear in the context.
3. If the context does not contain the answer, say that the provided documents do not cover it.
4. Be concise."""
MAX_ANSWER_TOKENS = 1024
PREVIEW_CHARS = 300
CHUNK_ID_RE = re.compile(r"^[0-9a-f]{8}-v\d+-\d+$")
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


CENTROIDS = CentroidCache()
CHANGED_VERDICTS = {"regenerated", "repaired", "replaced", "redacted"}
WITHHELD = "[withheld: PII redaction was unavailable]"


@dataclass
class Generation:
    """Q8 output handed to Q9."""

    answer: str
    messages: list[dict[str, str]] | None  # None: LLM not called (no context)
    context: list["Candidate"]
    llm: LLMProvider | None


@dataclass
class QueryDeps:
    settings: Settings
    bus: EventBus
    vector_store: VectorStore
    embedder: Embedder
    reranker: Reranker
    llm_factory: Callable[[], LLMProvider]
    graph_store: GraphStore


@dataclass
class Candidate:
    chunk_id: str
    text: str
    similarity: float
    metadata: dict[str, Any]
    collapsed_from: list[str] = field(default_factory=list)
    filtered: str | None = None  # reason the candidate was dropped at Q4
    rerank_score: float | None = None

    def view(self) -> dict[str, Any]:
        text = self.text if len(self.text) <= PREVIEW_CHARS else self.text[:PREVIEW_CHARS] + "…"
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.metadata.get("document_id"),
            "filename": self.metadata.get("filename"),
            "page": self.metadata.get("page"),
            "page_end": self.metadata.get("page_end"),
            "cluster_id": self.metadata.get("cluster_id"),
            "similarity": round(self.similarity, 4),
            "collapsed_from": self.collapsed_from,
            "filtered": self.filtered,
            "text": text,
        }


def chroma_where(filters: QueryFilters | None) -> dict[str, Any] | None:
    clauses = []
    if filters and filters.document_ids:
        clauses.append({"document_id": {"$in": filters.document_ids}})
    if filters and filters.cluster_ids:
        clauses.append({"cluster_id": {"$in": filters.cluster_ids}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _matches(meta: dict[str, Any], filters: QueryFilters | None) -> bool:
    if filters is None:
        return True
    if filters.document_ids and meta.get("document_id") not in filters.document_ids:
        return False
    return not (filters.cluster_ids and meta.get("cluster_id") not in filters.cluster_ids)


def collapse_and_filter(
    hits: list[tuple[Neighbour, str]],
    canonicals: dict[str, tuple[dict[str, Any], str]],
    threshold: float,
    filters: QueryFilters | None = None,
) -> list[Candidate]:
    """Collapse near-duplicate hits onto their canonical chunk, then apply the threshold.

    ``hits`` are ordered by similarity (best first), so a group's first hit carries its best
    similarity. A duplicate whose canonical is missing or excluded by ``filters`` stands alone.
    Returns every candidate in rank order; dropped ones have ``filtered`` set.
    """
    groups: dict[str, Candidate] = {}
    for hit, text in hits:
        target = hit.metadata.get("is_duplicate_of") or ""
        record = canonicals.get(target) if target else None
        if record is not None and _matches(record[0], filters):
            cid, (meta, ctext) = target, record
        else:
            cid, meta, ctext = hit.chunk_id, hit.metadata, text
        if cid in groups:
            if hit.chunk_id != cid:  # the canonical itself may be hit after its duplicate
                groups[cid].collapsed_from.append(hit.chunk_id)
            continue
        groups[cid] = Candidate(cid, ctext, hit.similarity, meta)
        if cid != hit.chunk_id:
            groups[cid].collapsed_from.append(hit.chunk_id)
    out = list(groups.values())
    for c in out:
        if c.similarity < threshold:
            c.filtered = f"similarity {c.similarity:.2f} < threshold {threshold:.2f}"
    return out


def build_messages(
    query: str, context: list[Candidate], facts: list[dict[str, Any]] | None = None
) -> list[dict[str, str]]:
    blocks = []
    for c in context:
        page, page_end = c.metadata.get("page"), c.metadata.get("page_end")
        pages = f"p. {page}" if page == page_end else f"pp. {page}-{page_end}"
        blocks.append(f"[{c.chunk_id}] ({c.metadata.get('filename')}, {pages})\n{c.text}")
    user = "Context:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion: {query}"
    if facts:
        # Facts go first and carry no ids: citations must point at context chunks.
        lines = "\n".join(f"- {f['source']} | {f['relation']} | {f['target']}" for f in facts)
        user = (
            "Knowledge graph facts (supporting context; cite the chunks, not these):\n"
            f"{lines}\n\n{user}"
        )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_citations(answer: str, valid_ids: set[str]) -> tuple[list[str], list[str]]:
    """(valid cited ids in first-seen order, cited ids not present in the context)."""
    cited: list[str] = []
    unknown: list[str] = []
    for group in BRACKET_RE.findall(answer):
        for token in re.split(r"[,;\s]+", group.strip()):
            if token in valid_ids:
                if token not in cited:
                    cited.append(token)
            elif CHUNK_ID_RE.match(token) and token not in unknown:
                unknown.append(token)
    return cited, unknown


def latest_ingest_id() -> str | None:
    """Changes whenever an ingest finishes, so corpus-derived caches know to refresh."""
    with Session(get_engine()) as s:
        row = s.exec(
            select(Run.id)
            .where(Run.kind == "ingest", Run.finished_at.is_not(None))  # type: ignore[union-attr]
            .order_by(Run.finished_at.desc())  # type: ignore[union-attr]
            .limit(1)
        ).first()
    return row


def _now() -> datetime:
    return datetime.now(UTC)


class QueryJob:
    def __init__(
        self,
        run_id: str,
        request: QueryRequest,
        params: QueryParams,
        deps: QueryDeps,
        guardrails: GuardrailSettings | None = None,
        pii_counts: dict[str, int] | None = None,
        online_eval: bool = True,
        precision_judge: bool = True,
    ):
        """``request.query`` must already be PII-masked (done by the API before persisting).

        ``online_eval`` runs Q10; ``precision_judge`` lets Q10 call the context-precision judge
        (offline eval turns it off and judges precision against its ground truth instead).
        """
        self.run_id = run_id
        self.request = request
        self.params = params
        self.guardrails = guardrails or GuardrailSettings()
        self.pii_counts = pii_counts or {}
        self.online_eval = online_eval
        self.precision_judge = precision_judge
        self.guard_log: dict[str, list[dict[str, Any]]] = {}
        self.rec = StageRecorder(deps.bus, run_id)
        self.meter = UsageMeter(lambda: self.rec.current)
        self.deps = replace(deps, llm_factory=lambda: MeteredLLM(deps.llm_factory(), self.meter))
        self.q9: OutputResult | None = None
        self.t0 = time.perf_counter()

    def run(self) -> None:
        """Execute the run; always ends with a terminal ``done`` message and a finished Run row."""
        run_id_var.set(self.run_id)
        active_meter.set(self.meter)
        self.t0 = time.perf_counter()
        status, result = "error", {}
        try:
            for stage_id in QUERY_STAGES:
                self.rec.emit(stage_id, "pending", summary="queued")
            result = self._pipeline()
            status = result.pop("status", "success")
        except Exception as exc:
            logger.exception("query failed")
            result = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            with Session(get_engine()) as s:
                run = s.get(Run, self.run_id)
                if run is not None:
                    run.status, run.result, run.finished_at = status, result, _now()
                    s.add(run)
                    s.commit()
            self.deps.bus.close(self.run_id, {"status": status, **result})

    def _skip(self, stage_id: QueryStageId, reason: str) -> None:
        self.rec.emit(stage_id, "success", duration_ms=0, summary=reason, data={"skipped": True})

    def _pipeline(self) -> dict[str, Any]:
        self._q1_validate()
        report = self._q2_input_guardrail()
        if report.blocked:
            reasons = "; ".join(f"{c.name} ({c.reason})" for c in report.blocked)
            return {
                "status": "blocked",
                "answer": BLOCKED_ANSWER.format(reasons=reasons),
                "citations": [],
                "blocked_by": [c.name for c in report.blocked],
                "guardrails": self.guard_log,
            }
        vector = self._q3_embed()
        candidates = self._q4_retrieve(vector)
        graph = self._q5_graph_retrieve(vector)
        fused = self._q6_fusion(candidates, graph, vector)
        context = self._q7_rerank(fused)
        generation = self._q8_generate(context, graph.triples if graph else [])
        result = self._q9_output_guardrail(generation)
        answer_ms = round((time.perf_counter() - self.t0) * 1000)
        self.deps.bus.publish(self.run_id, "answer", result)
        result["eval"] = self._q10_eval(generation, result["answer"], answer_ms)
        return result

    # ------------------------------------------------------------------ Q2
    def _off_topic_similarity(self) -> float | None:
        vs, emb = self.deps.vector_store, self.deps.embedder
        if vs.count() == 0:
            return None
        centroids = CENTROIDS.get(
            (vs.collection_name, vs.count(), latest_ingest_id()), vs.cluster_centroids
        )
        if len(centroids) == 0:
            return None
        return float(np.max(centroids @ emb.embed_query(self.request.query)))

    def _q2_input_guardrail(self) -> InputReport:
        g = self.guardrails.input
        params = {
            "length_max_tokens": g.length_check.max_tokens,
            "prompt_injection_threshold": g.prompt_injection.threshold,
            "toxicity_threshold": g.toxicity.threshold,
            "off_topic_threshold": g.off_topic.threshold,
        }
        with self.rec.stage("Q2_input_guardrail", params) as st:
            tokens = get_tokenizer(self.deps.settings.tokenizer_encoding).count(self.request.query)
            report = run_input_guards(
                self.request.query,
                tokens,
                self.pii_counts,
                g,
                self.deps.llm_factory,
                self.params.seed,
                self._off_topic_similarity,
            )
            checks = [c.as_dict() for c in report.checks]
            self.guard_log["input"] = checks
            st.data = {"checks": checks, "query_used": self.request.query}
            ran = [c for c in report.checks if c.verdict != "skipped"]
            if report.blocked:
                st.status = "blocked"
                st.summary = "blocked by " + ", ".join(c.name for c in report.blocked)
            elif report.warnings:
                st.status = "warning"
                st.summary = "warning: " + "; ".join(c.reason for c in report.warnings)
            else:
                masked = sum(c.verdict == "mask" for c in report.checks)
                st.summary = f"passed {len(ran)} check(s)" + (", PII masked" if masked else "")
        return report

    # ------------------------------------------------------------------ Q1
    def _q1_validate(self) -> None:
        # The envelope was validated synchronously by POST /api/query (invalid requests get a
        # 422 with field-level errors and never start a run); this stage records what was used.
        with self.rec.stage("Q1_validate", self.params.model_dump()) as st:
            st.summary = "request valid"
            st.data = {"request": self.request.model_dump(mode="json")}

    # ------------------------------------------------------------------ Q3
    def _q3_embed(self):  # noqa: ANN202
        emb = self.deps.embedder
        params = {"model": emb.model_name}
        with self.rec.stage("Q3_query_embed", params) as st:
            tokens = get_tokenizer(self.deps.settings.tokenizer_encoding).count(self.request.query)
            was_loaded = emb.loaded
            t0 = time.perf_counter()
            vector = emb.embed_query(self.request.query)
            ms = round((time.perf_counter() - t0) * 1000)
            st.summary = f"{tokens} tokens, embedded in {ms} ms"
            if not was_loaded:
                st.summary += " (includes model load)"
            st.data = {
                "token_count": tokens,
                "embedding_ms": ms,
                "model_loaded_now": not was_loaded,
                "dimension": len(vector),
                "vector_preview": [round(float(x), 4) for x in vector[:8]],
            }
        return vector

    # ------------------------------------------------------------------ Q4
    def _q4_retrieve(self, vector) -> list[Candidate]:  # noqa: ANN001
        p, filters = self.params, self.request.filters
        params = {"top_k": p.top_k, "similarity_threshold": p.similarity_threshold}
        with self.rec.stage("Q4_vector_retrieve", params) as st:
            vs = self.deps.vector_store
            hits = vs.search(vector, p.top_k, self.deps.embedder.model_name, chroma_where(filters))
            targets = {h.metadata.get("is_duplicate_of") for h, _ in hits} - {"", None}
            canonicals = vs.get_records(sorted(targets))
            candidates = collapse_and_filter(hits, canonicals, p.similarity_threshold, filters)
            kept = [c for c in candidates if c.filtered is None]
            collapsed = sum(len(c.collapsed_from) for c in candidates)
            st.data = {
                "candidates": [c.view() for c in candidates],
                "fetched": len(hits),
                "kept": len(kept),
                "filtered": len(candidates) - len(kept),
                "collapsed": collapsed,
            }
            if not hits:
                st.status, st.summary = "warning", "no chunks in the collection match"
            elif not kept:
                st.status = "warning"
                st.summary = f"all {len(candidates)} candidates below the similarity threshold"
            else:
                st.summary = f"{len(kept)} of {len(candidates)} candidates kept" + (
                    f", {collapsed} duplicate(s) collapsed" if collapsed else ""
                )
        return kept

    # ------------------------------------------------------------------ Q5
    def _q5_graph_retrieve(self, vector) -> GraphResult | None:  # noqa: ANN001
        hops, store = self.params.graph_hops, self.deps.graph_store
        if hops == 0:
            self._skip("Q5_graph_retrieve", "graph_hops = 0: knowledge-graph retrieval disabled")
            return None
        params = {"graph_hops": hops, "graph_backend": store.backend}
        with self.rec.stage("Q5_graph_retrieve", params) as st:
            # The graph is an enhancement: if its store is down (e.g. Neo4j unreachable) the
            # query continues on vector results instead of failing.
            try:
                if store.stats()["vertices"] == 0:
                    st.summary = "knowledge graph is empty (build it during ingest)"
                    st.data = {"skipped": True}
                    return None
                g = graph_retriever.retrieve(
                    store,
                    self.deps.embedder,
                    self.deps.llm_factory,
                    self.request.query,
                    vector,
                    hops,
                    self.params.seed,
                )
            except Exception as exc:
                logger.warning("graph retrieval failed", exc_info=True)
                st.status = "warning"
                error = f"{type(exc).__name__}: {exc}"[:300]
                st.summary = f"graph store unavailable ({error}); continuing with vector results"
                st.data = {"error": error}
                return None
            matched = {m.key for m in g.matches}
            st.data = {
                "query_entities": g.query_entities,
                "extraction_note": g.extraction_note,
                "match_threshold": graph_retriever.ENTITY_MATCH_MIN,
                "matches": [
                    {"id": m.key, "method": m.method, "score": round(m.score, 4), "probe": m.probe}
                    for m in g.matches
                ],
                "subgraph": {
                    "vertices": [
                        v.summary()
                        | {"hop": g.subgraph.hops.get(v.id, 0), "matched": v.id in matched}
                        for v in g.subgraph.vertices
                    ],
                    "edges": [e.summary() for e in g.subgraph.edges],
                },
                "evidence": [
                    {"chunk_id": cid, "score": score, "via": via}
                    for cid, score, via in g.evidence[:50]
                ],
                "triples": g.triples,
            }
            if not g.matches:
                st.summary = "no graph entities matched the query"
            else:
                st.summary = (
                    f"{len(g.matches)} entity match(es), {len(g.subgraph.vertices)} vertices "
                    f"within {hops} hop(s), {len(g.evidence)} evidence chunk(s)"
                )
        return g

    # ------------------------------------------------------------------ Q6
    def _q6_fusion(
        self,
        vector_hits: list[Candidate],
        graph: GraphResult | None,
        vector,  # noqa: ANN001
    ) -> list[Candidate]:
        if graph is None or not graph.evidence:
            self._skip("Q6_fusion", "no graph candidates: vector results passed through")
            return vector_hits
        p, vs = self.params, self.deps.vector_store
        w = p.hybrid_weight_vector
        params = {"hybrid_weight_vector": w, "rrf_k": RRF_K, "top_k": p.top_k}
        with self.rec.stage("Q6_fusion", params) as st:
            # Graph evidence: the top_k chunks that still exist and pass the request filters.
            ids = [cid for cid, _, _ in graph.evidence]
            records = self.deps.vector_store.get_records(ids)
            via = {cid: v for cid, _, v in graph.evidence}
            graph_ids = [
                cid
                for cid in ids
                if cid in records and _matches(records[cid][0], self.request.filters)
            ][: p.top_k]
            by_id = {c.chunk_id: c for c in vector_hits}
            missing = [cid for cid in graph_ids if cid not in by_id]
            vectors = vs.get_vectors(missing)
            for cid in missing:  # graph-only: similarity shown for reference, not thresholded
                meta, text = records[cid]
                sim = float(np.dot(vector, vectors[cid])) if cid in vectors else 0.0
                by_id[cid] = Candidate(cid, text, sim, meta)
            rows = weighted_rrf([c.chunk_id for c in vector_hits], graph_ids, w)
            # Only the top_k fused chunks go on to Q7: cross-encoder cost grows linearly with
            # candidates (~0.7 s per 512-token chunk on CPU), so fusion must not double it.
            fused = [by_id[r.chunk_id] for r in rows[: p.top_k]]
            st.data = {
                "rows": [
                    {
                        "chunk_id": r.chunk_id,
                        "filename": by_id[r.chunk_id].metadata.get("filename"),
                        "page": by_id[r.chunk_id].metadata.get("page"),
                        "source": r.source,
                        "vector_rank": r.vector_rank,
                        "graph_rank": r.graph_rank,
                        "fused_rank": i,
                        "fused_score": round(r.score, 6),
                        "via": via.get(r.chunk_id, []),
                        "kept": i <= p.top_k,
                    }
                    for i, r in enumerate(rows, start=1)
                ]
            }
            counts = {k: sum(r.source == k for r in rows) for k in ("vector", "graph", "both")}
            st.summary = (
                f"{len(rows)} fused: {counts['both']} both, {counts['vector']} vector-only, "
                f"{counts['graph']} graph-only (vector weight {w}); top {len(fused)} to re-ranking"
            )
        return fused

    # ------------------------------------------------------------------ Q7
    def _q7_rerank(self, candidates: list[Candidate]) -> list[Candidate]:
        top_n = self.params.top_n
        if not candidates:
            self._skip("Q7_rerank", "no candidates to re-rank")
            return []
        rr = self.deps.reranker
        params = {"top_n": top_n, "model": rr.model_name, "max_pair_tokens": MAX_PAIR_TOKENS}
        with self.rec.stage("Q7_rerank", params) as st:
            was_loaded = rr.loaded
            scores = rr.score(self.request.query, [c.text for c in candidates])
            for c, s in zip(candidates, scores, strict=True):
                c.rerank_score = s
            before = {c.chunk_id: i for i, c in enumerate(candidates, start=1)}
            ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
            st.data = {
                "ranking": [
                    {
                        "chunk_id": c.chunk_id,
                        "filename": c.metadata.get("filename"),
                        "page": c.metadata.get("page"),
                        "before_rank": before[c.chunk_id],
                        "after_rank": i,
                        "similarity": round(c.similarity, 4),
                        "score": round(c.rerank_score, 4),
                        "kept": i <= top_n,
                    }
                    for i, c in enumerate(ranked, start=1)
                ],
                "model_loaded_now": not was_loaded,
            }
            kept = ranked[:top_n]
            moved = sum(1 for i, c in enumerate(kept, start=1) if before[c.chunk_id] != i)
            st.summary = f"kept top {len(kept)} of {len(ranked)}; {moved} changed position"
            if not was_loaded:
                st.summary += " (includes model load)"
        return kept

    # ------------------------------------------------------------------ Q8
    def _redact(self, text: str) -> str:
        """PII-masked copy of pre-Q9 LLM text for persisting (when pii_leak is on). The live
        token stream is not persisted; stage events and the run record are."""
        if not self.guardrails.output.pii_leak.enabled or not text:
            return text
        try:
            return get_pii_engine().mask(text)[0]
        except Exception:  # never store what could not be checked
            logger.warning("PII redaction failed; withholding stored text", exc_info=True)
            return WITHHELD

    def _q8_generate(self, context: list[Candidate], facts: list[dict[str, Any]]) -> Generation:
        p, bus = self.params, self.deps.bus
        params = {"temperature": p.temperature, "seed": p.seed}
        with self.rec.stage("Q8_generate", params) as st:
            if not context:
                bus.publish(self.run_id, "token", {"text": NO_CONTEXT_ANSWER})
                st.status = "warning"
                st.summary = "no context passed retrieval; LLM not called"
                st.data = {"answer": NO_CONTEXT_ANSWER, "citations": [], "llm_called": False}
                return Generation(NO_CONTEXT_ANSWER, None, [], None)

            messages = build_messages(self.request.query, context, facts)
            llm = self.deps.llm_factory()
            result = llm.stream(
                messages,
                lambda delta: bus.publish(self.run_id, "token", {"text": delta}),
                temperature=p.temperature,
                seed=p.seed,
                max_tokens=MAX_ANSWER_TOKENS,
            )
            by_id = {c.chunk_id: c for c in context}
            cited, unknown = parse_citations(result.text, set(by_id))
            # Preview text only: the full chunk is already in ``prompt`` and served by
            # GET /api/chunks/{chunk_id}; repeating it could push Q8 past the SSE payload cap.
            citations = [
                {k: v for k, v in by_id[cid].view().items() if k != "filtered"} for cid in cited
            ]
            st.data = {
                "prompt": messages,
                "context_chunk_ids": list(by_id),
                "answer": self._redact(result.text),
                "citations": citations,
                "unknown_citations": unknown,
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "llm_called": True,
                "facts_used": len(facts),
            }
            st.summary = (
                f"{result.completion_tokens} tokens from {result.model}, {len(cited)} citation(s)"
            )
            if unknown:
                st.status = "warning"
                st.summary += f"; {len(unknown)} cited id(s) not in context"
            return Generation(result.text, messages, context, llm)

    # ------------------------------------------------------------------ Q9
    def _citations(self, answer: str, context: list[Candidate]) -> list[dict[str, Any]]:
        by_id = {c.chunk_id: c for c in context}
        cited, _ = parse_citations(answer, set(by_id))
        return [{k: v for k, v in by_id[cid].view().items() if k != "filtered"} for cid in cited]

    def _q9_output_guardrail(self, gen: Generation) -> dict[str, Any]:
        g = self.guardrails.output
        params = {
            "max_unsupported_ratio": g.groundedness.max_unsupported_ratio,
            "max_chars": g.format_check.max_chars,
        }
        with self.rec.stage("Q9_output_guardrail", params) as st:
            if gen.llm is None or gen.messages is None:
                checks = [
                    GuardCheck(name, "skipped", "no answer was generated").as_dict()
                    for name in ("citation_check", "groundedness", "pii_leak", "format_check")
                ]
                checks.insert(
                    3,
                    GuardCheck(
                        "no_answer_handling", "pass", "no context: standard not-found message"
                    ).as_dict(),
                )
                self.guard_log["output"] = checks
                st.summary = "no answer generated; standard not-found message"
                st.data = {"checks": checks, "modified": False, "regenerated": False}
                return {"answer": gen.answer, "citations": [], "guardrails": self.guard_log}

            res = run_output_guards(
                gen.answer,
                gen.messages,
                [c.chunk_id for c in gen.context],
                gen.llm,
                g,
                get_pii_engine(),
                parse_citations,
                self.params.temperature,
                self.params.seed,
            )
            self.q9 = res
            checks = [c.as_dict() for c in res.checks]
            for c in checks:  # unsupported sentences quote the unredacted answer
                if "unsupported" in c.get("details", {}):
                    c["details"]["unsupported"] = [
                        self._redact(t) for t in c["details"]["unsupported"]
                    ]
            self.guard_log["output"] = checks
            modified = res.answer != gen.answer
            original = self._redact(gen.answer)
            st.data = {
                "checks": checks,
                "modified": modified,
                "regenerated": res.regenerated,
                "original_answer": original,
                "original_redacted": original != gen.answer,
                "final_answer": res.answer,
                "judge": [j | {"text": self._redact(j["text"])} for j in res.judge],
            }
            changed = [c for c in res.checks if c.verdict in CHANGED_VERDICTS]
            if changed:
                st.status = "warning"
                st.summary = "; ".join(f"{c.name}: {c.verdict}" for c in changed)
            else:
                ran = sum(c.verdict != "skipped" for c in res.checks)
                st.summary = f"passed {ran} check(s); answer unchanged"
            result: dict[str, Any] = {
                "answer": res.answer,
                "citations": self._citations(res.answer, gen.context),
                "guardrails": self.guard_log,
            }
            if modified:
                result["original_answer"] = original
            return result

    # ------------------------------------------------------------------ Q10
    def _q10_eval(self, gen: Generation, answer: str, answer_ms: int) -> dict[str, Any] | None:
        if not self.online_eval:
            self._skip("Q10_eval", "online evaluation is off (Settings)")
            return None
        params = {"precision_judge": self.precision_judge}
        with self.rec.stage("Q10_eval", params) as st:
            seed, q9 = self.params.seed, self.q9
            context_text = gen.messages[-1]["content"] if gen.messages else ""
            metrics: dict[str, dict[str, Any]] = {
                "faithfulness": judge.faithfulness(
                    gen.llm,
                    answer,
                    context_text,
                    seed,
                    q9.judge if q9 else None,
                    q9.judged_text if q9 else None,
                )
            }
            if answer == NOT_FOUND:
                metrics["answer_relevancy"] = {"value": None, "note": "standard not-found answer"}
            else:
                value = answer_relevancy(self.deps.embedder, self.request.query, answer)
                metrics["answer_relevancy"] = {"value": round(value, 4), "note": ""}
            if not self.precision_judge:
                metrics["context_precision"] = {"value": None, "note": "judged by the eval run"}
            elif gen.llm is None:
                metrics["context_precision"] = {"value": None, "note": "no context chunks"}
            else:
                chunks = [(c.chunk_id, c.text) for c in gen.context]
                metrics["context_precision"] = judge.context_precision(
                    gen.llm, self.request.query, chunks, seed
                )
            out = {
                "metrics": metrics,
                "latency_ms": {"answer_ready": answer_ms, "stages": dict(self.rec.durations)},
                "tokens": self.meter.summary(),
            }
            st.data = out
            failed = [k for k, m in metrics.items() if "judge unavailable" in m.get("note", "")]
            scored = [f"{k} {m['value']:.2f}" for k, m in metrics.items() if m["value"] is not None]
            total = out["tokens"]["total"]
            st.summary = (", ".join(scored) or "no metric applies") + (
                f"; {total['calls']} LLM call(s), "
                f"{total['prompt_tokens'] + total['completion_tokens']:,} tokens"
            )
            if failed:
                st.status = "warning"
                st.summary += f"; judge unavailable for {', '.join(failed)}"
        return out
