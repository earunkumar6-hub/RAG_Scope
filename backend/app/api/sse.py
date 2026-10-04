"""SSE responses for run event streams (shared by ingest and query endpoints).

* Live or recently finished runs stream from the in-memory bus, resuming after ``Last-Event-ID``.
* Runs no longer in memory (evicted / server restarted) replay their persisted stage events from
  SQLite, followed by a ``done`` message carrying the stored result.
* A client already caught up past ``done`` gets ``204 No Content``, which tells ``EventSource`` to
  stop reconnecting (clients should also call ``close()`` on ``done``).
"""

import json
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request, Response
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from app.core.events import EventBus
from app.db.models import Run, StageEventRecord
from app.db.session import get_engine

PING_SECONDS = 15


def _last_event_id(request: Request) -> int:
    raw = request.headers.get("last-event-id") or request.query_params.get("lastEventId") or "0"
    try:
        return max(int(raw), 0)
    except ValueError:
        return 0


def run_event_stream(request: Request, bus: EventBus, run_id: str, kind: str) -> Response:
    """Build the SSE (or 204/404) response for ``run_id``."""
    last_id = _last_event_id(request)
    if bus.has(run_id):
        final = bus.final_seq(run_id)
        if final is not None and last_id >= final:
            return Response(status_code=204)

        async def live() -> AsyncIterator[dict[str, str]]:
            async for msg in bus.subscribe(run_id, last_id):
                yield msg.as_sse()

        return EventSourceResponse(live(), ping=PING_SECONDS)

    with Session(get_engine()) as s:
        run = s.get(Run, run_id)
        if run is None or run.kind != kind:
            raise HTTPException(status_code=404, detail=f"Unknown {kind} run '{run_id}'")
        records = s.exec(
            select(StageEventRecord)
            .where(StageEventRecord.run_id == run_id)
            .order_by(StageEventRecord.seq)
        ).all()
        status, result = run.status, dict(run.result or {})
    done_seq = (records[-1].seq if records else 0) + 1
    if last_id >= done_seq:
        return Response(status_code=204)
    events = [
        {"id": str(r.seq), "event": "stage", "data": json.dumps(r.payload)}
        for r in records
        if r.seq > last_id
    ]
    events.append(
        {
            "id": str(done_seq),
            "event": "done",
            "data": json.dumps({"status": status, "replayed": True, **result}, default=str),
        }
    )
    return EventSourceResponse(iter(events))
