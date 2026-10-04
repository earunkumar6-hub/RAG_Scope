"""``GET /api/runs``: run history. A run's full stage trace replays via its ``/events`` endpoint."""

from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import Session, col, func, select

from app.db.models import Run
from app.db.session import get_session

router = APIRouter(prefix="/api/runs", tags=["runs"])

SessionDep = Annotated[Session, Depends(get_session)]


class RunSummary(BaseModel):
    id: str
    kind: str
    status: str
    label: str  # query text, or uploaded file names
    created_at: datetime
    finished_at: datetime | None


class RunPage(BaseModel):
    items: list[RunSummary]
    total: int


def _label(run: Run) -> str:
    if run.kind == "query":
        return str(run.request.get("query", ""))
    return ", ".join(run.request.get("files", []))


@router.get("", response_model=RunPage)
def list_runs(
    session: SessionDep,
    kind: Literal["ingest", "query"] | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> RunPage:
    """Runs newest first, optionally filtered by kind."""
    stmt = select(Run)
    if kind is not None:
        stmt = stmt.where(Run.kind == kind)
    total = session.exec(select(func.count()).select_from(stmt.subquery())).one()
    runs = session.exec(stmt.order_by(col(Run.created_at).desc()).offset(offset).limit(limit)).all()
    return RunPage(
        items=[
            RunSummary(
                id=r.id,
                kind=r.kind,
                status=r.status,
                label=_label(r),
                created_at=r.created_at,
                finished_at=r.finished_at,
            )
            for r in runs
        ],
        total=total,
    )
