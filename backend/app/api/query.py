"""``POST /api/query`` and ``GET /api/query/{run_id}/events`` (SSE)."""

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from pydantic import BaseModel
from sqlmodel import Session

from app.api.sse import run_event_stream
from app.core.events import bus
from app.db.models import Run
from app.db.session import get_engine
from app.guardrails.pii import get_pii_engine, summarize
from app.pipeline.query.runner import QueryDeps, QueryJob
from app.schemas.errors import ErrorDetail, JEVError
from app.schemas.params import QueryParams
from app.schemas.query import QueryRequest

router = APIRouter(prefix="/api/query", tags=["query"])


class QueryAccepted(BaseModel):
    run_id: str
    params: QueryParams


@router.post("", response_model=QueryAccepted, status_code=202)
def query(body: QueryRequest, request: Request, background: BackgroundTasks) -> QueryAccepted:
    """Validate the envelope (Q1), resolve params, then run Q1–Q10 in the background."""
    state = request.app.state
    if body.collection != state.vector_store.collection_name:
        raise JEVError(
            [
                ErrorDetail(
                    field="collection",
                    message=f"unknown collection; available: {state.vector_store.collection_name}",
                )
            ]
        )
    config = state.runtime_config.get()
    params = config.defaults.resolve_query(body.params)

    # PII is masked here, before anything is persisted, logged or retrieved: the job, the run
    # record and every stage event only ever see the masked query.
    pii_counts: dict[str, int] = {}
    if config.guardrails.input.pii_detection.enabled:
        try:
            masked, findings = get_pii_engine().mask(body.query)
        except Exception as exc:  # fail closed rather than persist unmasked text
            raise HTTPException(
                status_code=503,
                detail=f"PII detector unavailable ({type(exc).__name__}); disable "
                "pii_detection in Settings to query without it",
            ) from exc
        body = body.model_copy(update={"query": masked})
        pii_counts = summarize(findings)

    run_id = str(uuid.uuid4())
    with Session(get_engine()) as s:
        s.add(
            Run(
                id=run_id,
                kind="query",
                request={**body.model_dump(mode="json"), "params": params.model_dump()},
            )
        )
        s.commit()
    bus.open(run_id)
    deps = QueryDeps(
        settings=state.settings,
        bus=bus,
        vector_store=state.vector_store,
        embedder=state.embedder,
        reranker=state.reranker,
        llm_factory=state.llm_factory,
        graph_store=state.graph_store,
    )
    job = QueryJob(
        run_id,
        body,
        params,
        deps,
        config.guardrails,
        pii_counts,
        online_eval=config.evaluation.online.enabled,
    )
    background.add_task(job.run)
    return QueryAccepted(run_id=run_id, params=params)


@router.get("/{run_id}/events")
def query_events(run_id: str, request: Request) -> Response:
    """SSE stream of StageEvents (Q1–Q10), answer ``token`` messages, the final ``answer``
    (before Q10), then ``done``."""
    return run_event_stream(request, bus, run_id, kind="query")
