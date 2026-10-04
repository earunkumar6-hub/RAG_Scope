"""Deterministic, persistent cache for query-time LLM JSON calls (see ``LLMCache``)."""

import hashlib
from collections.abc import Callable
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.db.models import LLMCache
from app.db.session import get_engine
from app.llm.usage import active_meter


def normalise_text(text: str) -> str:
    return " ".join(text.casefold().split())


def cache_key(kind: str, model: str, text: str) -> str:
    raw = f"{kind}\x1f{model}\x1f{normalise_text(text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_json(
    kind: str, model: str, text: str, compute: Callable[[], dict[str, Any]]
) -> tuple[dict[str, Any], bool]:
    """Return (value, cache_hit). Only successful results are stored; errors propagate."""
    key = cache_key(kind, model, text)
    with Session(get_engine()) as s:
        row = s.get(LLMCache, key)
        if row is not None:
            if (meter := active_meter.get()) is not None:
                meter.cache_hit()
            return dict(row.value), True
    value = compute()
    with Session(get_engine()) as s:
        try:
            s.add(LLMCache(key=key, kind=kind, model=model, value=value))
            s.commit()
        except IntegrityError:  # a concurrent identical query stored it first
            s.rollback()
    return value, False
