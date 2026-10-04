"""Headless stand-in for ``EventBus`` used when eval runs the ingest/query pipelines."""

from typing import Any


class CaptureBus:
    """Keeps each stage's latest event and the final result; emits no SSE, persists nothing."""

    def __init__(self) -> None:
        self.stages: dict[str, dict[str, Any]] = {}
        self.done: dict[str, Any] = {}

    def publish_stage(self, event: Any) -> int:  # noqa: ANN401
        self.stages[event.stage_id] = event.model_dump(mode="json")
        return 0

    def publish(self, run_id: str, event: str, payload: dict[str, Any]) -> int:
        if event == "done":
            self.done = payload
        return 0

    def close(self, run_id: str, payload: dict[str, Any] | None = None) -> int:
        return self.publish(run_id, "done", payload or {})
