"""SQLModel tables."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class Run(SQLModel, table=True):
    """One ingestion job or query run; its stage trace lives in ``StageEventRecord``."""

    id: str = Field(primary_key=True)
    kind: str = Field(index=True)  # "ingest" | "query"
    status: str = Field(default="running", index=True)
    request: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)
    finished_at: datetime | None = None


class StageEventRecord(SQLModel, table=True):
    """Persisted copy of every ``StageEvent`` so past runs can be re-opened."""

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, foreign_key="run.id")
    seq: int
    stage_id: str = Field(index=True)
    status: str
    started_at: datetime
    duration_ms: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class Document(SQLModel, table=True):
    """An ingested source file. ``version`` increments each time it is re-chunked."""

    id: str = Field(primary_key=True)
    filename: str
    extension: str
    size_bytes: int
    sha256: str = Field(index=True)
    page_count: int = 0
    total_tokens: int = 0
    chunk_count: int = 0
    duplicate_chunk_count: int = 0
    chunk_size: int
    chunk_overlap: int
    version: int = 1
    status: str = Field(default="processing", index=True)  # processing | ready | error
    error: str | None = None
    last_job_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Chunk(SQLModel, table=True):
    """A chunk of a document; mirrors the Chroma record (text + metadata, no vector)."""

    id: str = Field(primary_key=True)  # "<doc8>-v<version>-<index>"
    document_id: str = Field(index=True, foreign_key="document.id")
    chunk_index: int
    page: int
    page_end: int
    text: str
    token_count: int
    start_offset: int
    end_offset: int
    overlap_prev_chars: int = 0
    cluster_id: int = Field(default=-1, index=True)
    is_duplicate_of: str | None = Field(default=None, index=True)
    duplicate_similarity: float | None = None


class Cluster(SQLModel, table=True):
    """Topic cluster over unique chunks of the whole corpus (recomputed on every ingest)."""

    id: int = Field(primary_key=True)
    label: str
    label_source: str  # "llm" | "keywords"
    size: int
    member_hash: str  # hash of sorted member chunk ids; unchanged membership keeps its label


class LLMCache(SQLModel, table=True):
    """Cached LLM JSON results for query-time steps (Q2 classifier, Q5 entity extraction, eval
    judges).

    Keyed by step + model + normalised input, so a repeated query gets identical scores and
    entities (LLM seeds are best-effort), and costs nothing the second time.
    """

    key: str = Field(primary_key=True)  # sha256(kind, model, normalised input)
    kind: str = Field(index=True)
    model: str
    value: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)


class EvalDataset(SQLModel, table=True):
    """A golden set: uploaded JSONL or synthetic Q&A generated from chunks."""

    id: str = Field(primary_key=True)
    name: str
    source: str  # "upload" | "synthetic"
    status: str = Field(default="ready", index=True)  # generating | ready | error
    error: str | None = None
    question_count: int = 0
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)


class EvalQuestion(SQLModel, table=True):
    """One golden question. ``relevant_spans``: [{sha256, start, end, chunk_id, filename, page}]
    in parsed-text offsets (see ``app.eval.metrics``)."""

    id: int | None = Field(default=None, primary_key=True)
    dataset_id: str = Field(index=True, foreign_key="evaldataset.id")
    idx: int
    question: str
    ground_truth: str
    relevant_spans: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))


class EvalRun(SQLModel, table=True):
    """A dataset evaluated over a grid of query parameters; ``cells`` are the resolved params
    per grid cell, ``summary`` the per-cell metric means."""

    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True, foreign_key="evaldataset.id")
    status: str = Field(default="running", index=True)  # running | success | error | cancelled
    param_grid: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    cells: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    total: int = 0
    done: int = 0
    summary: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow, index=True)
    finished_at: datetime | None = None


class EvalResult(SQLModel, table=True):
    """One question answered and scored under one grid cell."""

    id: int | None = Field(default=None, primary_key=True)
    eval_run_id: str = Field(index=True, foreign_key="evalrun.id")
    cell: int
    question_id: int = Field(foreign_key="evalquestion.id")
    status: str  # pipeline status: success | blocked | error
    answer: str = ""
    error: str | None = None
    retrieved: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    details: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    latency_ms: int = 0
    tokens: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class EvalIndex(SQLModel, table=True):
    """A throwaway index (own SQLite, Chroma and NetworkX graph under ``eval_indexes/<dir>``)
    of the corpus re-chunked at one chunk size. ``id`` hashes everything the build depends on,
    so a changed corpus or setting means a different index."""

    id: str = Field(primary_key=True)  # sha256 of corpus + ingest params + models
    dir_name: str
    chunk_size: int
    chunk_overlap: int
    dedup_threshold: float
    build_graph: bool
    corpus: list[str] = Field(default_factory=list, sa_column=Column(JSON))  # document sha256s
    embedder: str
    kg_model: str
    status: str = Field(default="building", index=True)  # building | ready | error
    error: str | None = None
    chunk_count: int = 0
    vertex_count: int = 0
    edge_count: int = 0
    build_tokens: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)
    finished_at: datetime | None = None
