"""In-process SSE event bus with per-run replay buffers.

Design:
* Each run (ingest job / query) gets a ``RunChannel`` holding every message it has emitted, so a
  client that connects after ``POST`` returned (or reconnects with ``Last-Event-ID``) misses
  nothing.
* ``publish`` is thread-safe: pipelines run CPU-bound work in worker threads and hand messages to
  the event loop via ``call_soon_threadsafe``. Sequence numbers are assigned under a lock in the
  same order the appends are scheduled, so ``seq`` always matches buffer position.
* Message types: ``stage`` (a ``StageEvent``), ``token`` (answer token), ``answer`` (the final
  query answer, sent before Q10 scores it), ``done`` (terminal; the stream closes after it is
  delivered).
* Stage events are persisted to SQLite as they are published, for run history.
"""

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlmodel import Session

from app.db.models import StageEventRecord
from app.db.session import get_engine
from app.schemas.events import StageEvent

logger = logging.getLogger(__name__)

MessageType = Literal["stage", "token", "answer", "done"]


@dataclass(frozen=True)
class BusMessage:
    seq: int
    event: MessageType
    data: str  # JSON-encoded payload

    def as_sse(self) -> dict[str, str]:
        return {"id": str(self.seq), "event": self.event, "data": self.data}


@dataclass
class RunChannel:
    messages: list[BusMessage] = field(default_factory=list)
    next_seq: int = 1
    closed: bool = False
    closed_at: float | None = None
    signal: asyncio.Event = field(default_factory=asyncio.Event)


class UnknownRunError(KeyError):
    pass


class EventBus:
    """Fan-out of run messages to any number of SSE subscribers, with full replay."""

    def __init__(self, retention_seconds: float = 3600.0):
        self._channels: dict[str, RunChannel] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._retention = retention_seconds

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the server event loop (called at startup)."""
        self._loop = loop

    def open(self, run_id: str) -> None:
        """Register a run before its id is returned to the client."""
        with self._lock:
            self._evict_expired()
            self._channels.setdefault(run_id, RunChannel())

    def has(self, run_id: str) -> bool:
        return run_id in self._channels

    def publish(self, run_id: str, event: MessageType, payload: dict[str, Any]) -> int:
        """Append a message to a run's channel from any thread. Returns its sequence number."""
        if self._loop is None:
            raise RuntimeError("EventBus is not bound to an event loop")
        data = json.dumps(payload, default=str)
        with self._lock:
            channel = self._channels.get(run_id)
            if channel is None:
                raise UnknownRunError(run_id)
            if channel.closed:
                raise RuntimeError(f"Run {run_id} is already closed")
            seq = channel.next_seq
            channel.next_seq += 1
            if event == "done":
                channel.closed = True  # reject further publishes immediately
            msg = BusMessage(seq=seq, event=event, data=data)
            self._loop.call_soon_threadsafe(self._append, channel, msg)
        return seq

    def publish_stage(self, event: StageEvent) -> int:
        """Persist a ``StageEvent`` then stream it."""
        payload = event.model_dump(mode="json")
        seq = self.publish(str(event.run_id), "stage", payload)
        with Session(get_engine()) as session:
            session.add(
                StageEventRecord(
                    run_id=str(event.run_id),
                    seq=seq,
                    stage_id=event.stage_id,
                    status=event.status,
                    started_at=event.started_at,
                    duration_ms=event.duration_ms,
                    payload=payload,
                )
            )
            session.commit()
        return seq

    def close(self, run_id: str, payload: dict[str, Any] | None = None) -> int:
        """Emit the terminal ``done`` message; subscribers finish after receiving it."""
        return self.publish(run_id, "done", payload or {})

    async def subscribe(self, run_id: str, after_seq: int = 0) -> AsyncIterator[BusMessage]:
        """Yield all messages with ``seq > after_seq``, then live ones until ``done``."""
        channel = self._channels.get(run_id)
        if channel is None:
            raise UnknownRunError(run_id)
        index = max(after_seq, 0)
        while True:
            signal = channel.signal
            while index < len(channel.messages):
                msg = channel.messages[index]
                index += 1
                yield msg
            if channel.messages and channel.messages[-1].event == "done":
                return
            await signal.wait()

    def final_seq(self, run_id: str) -> int | None:
        """Sequence number of the delivered ``done`` message, or None while the run is live."""
        channel = self._channels.get(run_id)
        if channel is None or not channel.messages or channel.messages[-1].event != "done":
            return None
        return channel.messages[-1].seq

    @staticmethod
    def _append(channel: RunChannel, msg: BusMessage) -> None:
        channel.messages.append(msg)
        if msg.event == "done":
            channel.closed_at = time.monotonic()
        previous, channel.signal = channel.signal, asyncio.Event()
        previous.set()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            rid
            for rid, ch in self._channels.items()
            if ch.closed_at is not None and now - ch.closed_at > self._retention
        ]
        for rid in expired:
            del self._channels[rid]


bus = EventBus()
