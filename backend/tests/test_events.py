import asyncio
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.core.events import EventBus, UnknownRunError
from app.db.models import Run, StageEventRecord
from app.db.session import get_engine, init_engine
from app.schemas.events import StageEvent


@pytest.fixture
async def bus(tmp_path: Path) -> EventBus:
    init_engine(tmp_path / "events.db")
    b = EventBus()
    b.bind_loop(asyncio.get_running_loop())
    return b


async def collect(bus: EventBus, run_id: str, after: int = 0) -> list:
    return [m async for m in bus.subscribe(run_id, after)]


async def drain_loop() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_late_subscriber_gets_full_replay(bus: EventBus):
    bus.open("r1")
    bus.publish("r1", "token", {"t": "a"})
    bus.publish("r1", "token", {"t": "b"})
    bus.close("r1", {"status": "success"})
    await drain_loop()
    msgs = await collect(bus, "r1")
    assert [(m.seq, m.event) for m in msgs] == [(1, "token"), (2, "token"), (3, "done")]


async def test_last_event_id_resumes_after_seq(bus: EventBus):
    bus.open("r2")
    for i in range(4):
        bus.publish("r2", "token", {"i": i})
    bus.close("r2")
    await drain_loop()
    msgs = await collect(bus, "r2", after=2)
    assert [m.seq for m in msgs] == [3, 4, 5]


async def test_live_publish_from_worker_thread_in_order(bus: EventBus):
    bus.open("r3")
    task = asyncio.create_task(collect(bus, "r3"))

    def worker() -> None:
        for i in range(50):
            bus.publish("r3", "token", {"i": i})
        bus.close("r3")

    thread = threading.Thread(target=worker)
    thread.start()
    msgs = await asyncio.wait_for(task, timeout=5)
    thread.join()
    assert [m.seq for m in msgs] == list(range(1, 52))
    assert msgs[-1].event == "done"


async def test_multiple_subscribers_and_publish_after_close_rejected(bus: EventBus):
    bus.open("r4")
    t1 = asyncio.create_task(collect(bus, "r4"))
    t2 = asyncio.create_task(collect(bus, "r4"))
    await asyncio.sleep(0)
    bus.publish("r4", "token", {"x": 1})
    bus.close("r4")
    a, b = await asyncio.wait_for(asyncio.gather(t1, t2), timeout=5)
    assert [m.seq for m in a] == [m.seq for m in b] == [1, 2]
    with pytest.raises(RuntimeError):
        bus.publish("r4", "token", {})


async def test_unknown_run(bus: EventBus):
    with pytest.raises(UnknownRunError):
        bus.publish("nope", "token", {})
    with pytest.raises(UnknownRunError):
        await collect(bus, "nope")


async def test_stage_events_are_persisted(bus: EventBus):
    run_id = uuid.uuid4()
    with Session(get_engine()) as s:
        s.add(Run(id=str(run_id), kind="query"))
        s.commit()
    bus.open(str(run_id))
    event = StageEvent(
        run_id=run_id,
        stage_id="Q1_validate",
        status="success",
        started_at=datetime.now(UTC),
        duration_ms=3,
        summary="valid",
    )
    seq = bus.publish_stage(event)
    with Session(get_engine()) as s:
        rows = s.exec(select(StageEventRecord).where(StageEventRecord.run_id == str(run_id))).all()
    assert len(rows) == 1
    assert rows[0].seq == seq and rows[0].stage_id == "Q1_validate"
    assert rows[0].payload["summary"] == "valid"
