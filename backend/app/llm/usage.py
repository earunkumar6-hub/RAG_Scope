"""Per-run LLM token accounting for Q10.

``MeteredLLM`` wraps a provider and records every completion against the stage that made it
(``stage_of()``, normally ``StageRecorder.current``). ``cached_json`` reports cache hits to the
meter in ``active_meter``, so cached steps show up as calls that cost 0 tokens.
"""

import threading
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from app.llm.base import LLMProvider, LLMResult


class UsageMeter:
    def __init__(self, stage_of: Callable[[], str | None]):
        self._stage_of = stage_of
        self._lock = threading.Lock()  # KG extraction records from worker threads
        self.by_stage: dict[str, dict[str, int]] = {}

    def _row(self) -> dict[str, int]:
        stage = self._stage_of() or "other"
        return self.by_stage.setdefault(
            stage, {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )

    def record(self, result: LLMResult) -> None:
        with self._lock:
            row = self._row()
            row["calls"] += 1
            row["prompt_tokens"] += result.prompt_tokens
            row["completion_tokens"] += result.completion_tokens

    def cache_hit(self) -> None:
        with self._lock:
            self._row()["cache_hits"] += 1

    def summary(self) -> dict[str, Any]:
        total = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for row in self.by_stage.values():
            for k, v in row.items():
                total[k] += v
        return {"total": total, "by_stage": self.by_stage}


active_meter: ContextVar[UsageMeter | None] = ContextVar("active_meter", default=None)


class MeteredLLM(LLMProvider):
    """Delegates to ``inner`` and records token usage on ``meter``."""

    def __init__(self, inner: LLMProvider, meter: UsageMeter):
        self.inner, self.meter = inner, meter
        self.name, self.model, self.fast_model = inner.name, inner.model, inner.fast_model

    def complete(self, messages, **kwargs: Any) -> LLMResult:  # noqa: ANN001
        result = self.inner.complete(messages, **kwargs)
        self.meter.record(result)
        return result

    def stream(self, messages, on_token, **kwargs: Any) -> LLMResult:  # noqa: ANN001
        result = self.inner.stream(messages, on_token, **kwargs)
        self.meter.record(result)
        return result
