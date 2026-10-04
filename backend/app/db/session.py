"""SQLite engine/session management.

``use_engine`` scopes an override of ``get_engine()`` to the current context. Eval runs use it to
run the unchanged ingest and query pipelines against a throwaway index's own database.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.db import models  # noqa: F401  (register tables)

_engine: Engine | None = None
_override: ContextVar[Engine | None] = ContextVar("engine_override", default=None)


def create_db(sqlite_path: Path) -> Engine:
    """An engine for ``sqlite_path`` with every table created."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{sqlite_path.as_posix()}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    return engine


def init_engine(sqlite_path: Path) -> Engine:
    """Create the engine and tables. Called once at app startup."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = create_db(sqlite_path)
    return _engine


def get_engine() -> Engine:
    if (override := _override.get()) is not None:
        return override
    if _engine is None:
        raise RuntimeError("Database engine not initialised")
    return _engine


@contextmanager
def use_engine(engine: Engine | None) -> Iterator[None]:
    """Within this block (same thread/context), ``get_engine()`` returns ``engine``."""
    token = _override.set(engine)
    try:
        yield
    finally:
        _override.reset(token)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session."""
    with Session(get_engine()) as session:
        yield session
