"""Stage instrumentation shared by the ingestion and query pipelines.

``StageRecorder.stage(...)`` emits a ``running`` event on entry and a terminal event (status chosen
by the stage body, ``error`` on exception) with duration, params, summary and capped data on exit.
"""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.core.events import EventBus
from app.schemas.events import StageEvent, StageId, StageStatus

logger = logging.getLogger(__name__)

MAX_DATA_BYTES = 256_000


@dataclass
class StageContext:
    """Mutable outcome of a stage, filled in by the stage body."""

    status: StageStatus = "success"
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def cap_data(data: dict[str, Any]) -> dict[str, Any]:
    """Keep SSE payloads small; oversized payloads are replaced by a marker (fetch via REST)."""
    size = len(json.dumps(data, default=str))
    if size <= MAX_DATA_BYTES:
        return data
    return {"truncated": True, "size_bytes": size, "keys": sorted(data)}


class StageRecorder:
    """Emits ``StageEvent``s for one run."""

    def __init__(self, bus: EventBus, run_id: str):
        self.bus = bus
        self.run_id = UUID(run_id)
        self.current: StageId | None = None  # stage whose body is running (for LLM metering)
        self.durations: dict[str, int] = {}  # terminal duration per stage, in ms

    def emit(
        self,
        stage_id: StageId,
        status: StageStatus,
        *,
        started_at: datetime | None = None,
        duration_ms: int | None = None,
        params_used: dict[str, Any] | None = None,
        summary: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        if duration_ms is not None:
            self.durations[stage_id] = duration_ms
        self.bus.publish_stage(
            StageEvent(
                run_id=self.run_id,
                stage_id=stage_id,
                status=status,
                started_at=started_at or datetime.now(UTC),
                duration_ms=duration_ms,
                params_used=params_used or {},
                summary=summary,
                data=cap_data(data or {}),
            )
        )

    @contextmanager
    def stage(
        self, stage_id: StageId, params_used: dict[str, Any] | None = None, summary: str = ""
    ) -> Iterator[StageContext]:
        """Run a stage body, emitting running + terminal events around it."""
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()
        self.emit(
            stage_id,
            "running",
            started_at=started_at,
            params_used=params_used,
            summary=summary or "running",
        )
        ctx = StageContext()
        self.current = stage_id
        try:
            yield ctx
        except Exception as exc:
            ms = round((time.perf_counter() - t0) * 1000)
            logger.exception("stage failed", extra={"stage_id": stage_id})
            self.emit(
                stage_id,
                "error",
                started_at=started_at,
                duration_ms=ms,
                params_used=params_used,
                summary=f"{type(exc).__name__}: {exc}"[:500],
                data=ctx.data,
            )
            raise
        finally:
            self.current = None
        ms = round((time.perf_counter() - t0) * 1000)
        self.emit(
            stage_id,
            ctx.status,
            started_at=started_at,
            duration_ms=ms,
            params_used=params_used,
            summary=ctx.summary,
            data=ctx.data,
        )

    def progress(self, stage_id: StageId, summary: str, data: dict[str, Any]) -> None:
        """Intermediate ``running`` event (e.g. embedding batch progress)."""
        self.emit(stage_id, "running", summary=summary, data=data)
