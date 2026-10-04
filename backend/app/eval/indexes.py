"""Throwaway per-chunk_size indexes for eval runs whose grid includes chunk_size.

Each index re-ingests the live corpus (the kept original uploads) at one chunk size into its own
directory: own SQLite (via ``use_engine``), own Chroma collection, own NetworkX graph (never
Neo4j). It runs the unchanged ``IngestionJob``, so dedup, clustering and the KG build match a
real ingest. Every grid cell of such a run, including one matching the live settings, queries
an index built this way, so cells differ only in their parameters.

Indexes are cached by a hash of everything the build depends on (corpus file hashes, ingest
parameters, ingest seed, embedder, KG model); a changed corpus simply means a new key. Each build
gets a fresh directory name: Chroma keeps one client per path for the life of the process, so a
deleted index's path is never reused.
"""

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, col, func, select

from app.core.config import Settings
from app.db.models import Chunk, Document, EvalIndex
from app.db.session import create_db, use_engine
from app.eval.capture import CaptureBus
from app.graph.networkx_store import NetworkXGraphStore
from app.llm.usage import MeteredLLM, UsageMeter, active_meter
from app.pipeline.ingestion.chunker import chunk_text
from app.pipeline.ingestion.parser import parse_document
from app.pipeline.ingestion.runner import IngestDeps, IngestionJob
from app.pipeline.ingestion.tokenizer import get_tokenizer
from app.pipeline.ingestion.upload import UploadedFile
from app.pipeline.ingestion.vector_store import VectorStore
from app.pipeline.query.runner import QueryDeps
from app.schemas.params import IngestParams

logger = logging.getLogger(__name__)


@dataclass
class Shadow:
    """An opened index: what a query against it needs."""

    key: str
    engine: Engine
    vector_store: VectorStore
    graph_store: NetworkXGraphStore

    @property
    def collection(self) -> str:
        return self.vector_store.collection_name


_open: dict[str, Shadow] = {}  # by EvalIndex.dir_name (unique per build)


def _now() -> datetime:
    return datetime.now(UTC)


def corpus(s: Session) -> list[Document]:
    """The live corpus: ready documents, in a stable order."""
    return list(
        s.exec(select(Document).where(Document.status == "ready").order_by(col(Document.sha256)))
    )


def missing_sources(docs: list[Document], settings: Settings) -> list[str]:
    """Filenames whose original upload was not kept (ingested before uploads were saved)."""
    return [d.filename for d in docs if not (settings.uploads_dir / d.sha256).exists()]


def kg_model(deps: QueryDeps) -> str:
    try:
        return deps.llm_factory().fast_model
    except Exception:  # no LLM configured: the build will skip the graph
        return "none"


def index_key(shas: list[str], params: IngestParams, seed: int, embedder: str, model: str) -> str:
    raw = json.dumps(
        {
            "corpus": sorted(shas),
            "params": params.model_dump(),
            "seed": seed,
            "embedder": embedder,
            "kg_model": model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _paths(settings: Settings, dir_name: str) -> Path:
    return settings.eval_indexes_dir / dir_name


def _open_index(row: EvalIndex, settings: Settings) -> Shadow:
    shadow = _open.get(row.dir_name)
    if shadow is None:
        path = _paths(settings, row.dir_name)
        shadow = Shadow(
            row.id,
            create_db(path / "graphrag.db"),
            VectorStore(path / "chroma", f"idx-{row.dir_name}"),
            NetworkXGraphStore(path),
        )
        _open[row.dir_name] = shadow
    return shadow


_chunk_counts: dict[tuple[str, int, int], int] = {}  # (sha256, size, overlap) -> chunks


def count_chunks(docs: list[Document], settings: Settings, chunk_size: int, overlap: int) -> int:
    """Exact chunk count at ``chunk_size``: parses and chunks the kept uploads (CPU only, cached
    per file). A document without its original falls back to a token-based estimate."""
    tok = get_tokenizer(settings.tokenizer_encoding)
    total = 0
    for d in docs:
        key = (d.sha256, chunk_size, overlap)
        if key not in _chunk_counts:
            raw = settings.uploads_dir / d.sha256
            if not raw.exists():
                step = max(1, chunk_size - overlap)
                total += max(1, -(-max(0, d.total_tokens - overlap) // step))
                continue
            text = parse_document(d.extension, raw.read_bytes()).text
            _chunk_counts[key] = len(chunk_text(text, tok, chunk_size, overlap))
        total += _chunk_counts[key]
    return total


def ensure(
    params: IngestParams,
    seed: int,
    deps: QueryDeps,
    settings: Settings,
    engine: Engine,
) -> Shadow:
    """The ready index for ``params`` over the current corpus, building it if needed.

    ``engine`` is the live database (index rows live there). Raises ``RuntimeError`` if the
    build fails."""
    with Session(engine) as s:
        docs = corpus(s)
        model = kg_model(deps)
        key = index_key([d.sha256 for d in docs], params, seed, deps.embedder.model_name, model)
        row = s.get(EvalIndex, key)
        if row is not None and row.status == "ready" and _paths(settings, row.dir_name).exists():
            return _open_index(row, settings)
        if row is not None:  # failed or interrupted earlier: rebuild in a fresh directory
            _remove_dir(settings, row)
            s.delete(row)
            s.commit()
        row = EvalIndex(
            id=key,
            dir_name=f"{key[:12]}-{uuid.uuid4().hex[:6]}",
            chunk_size=params.chunk_size,
            chunk_overlap=params.chunk_overlap,
            dedup_threshold=params.dedup_threshold,
            build_graph=params.build_graph,
            corpus=[d.sha256 for d in docs],
            embedder=deps.embedder.model_name,
            kg_model=model,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        files = [
            UploadedFile(d.filename, d.extension, (settings.uploads_dir / d.sha256).read_bytes())
            for d in docs
        ]
    try:
        shadow = _build(row, files, params, seed, deps, settings, engine)
    except Exception as exc:
        with Session(engine) as s:
            failed = s.get(EvalIndex, key)
            if failed is not None:
                failed.status, failed.error = "error", f"{type(exc).__name__}: {exc}"[:500]
                failed.finished_at = _now()
                s.add(failed)
                s.commit()
        _open.pop(row.dir_name, None)
        raise RuntimeError(
            f"building the chunk_size={params.chunk_size} index failed: {exc}"
        ) from exc
    return shadow


def _build(
    row: EvalIndex,
    files: list[UploadedFile],
    params: IngestParams,
    seed: int,
    deps: QueryDeps,
    settings: Settings,
    engine: Engine,
) -> Shadow:
    shadow = _open_index(row, settings)
    meter = UsageMeter(lambda: "index_build")
    bus = CaptureBus()
    ingest = IngestDeps(
        settings=settings,
        bus=bus,
        vector_store=shadow.vector_store,
        embedder=deps.embedder,
        llm_factory=lambda: MeteredLLM(deps.llm_factory(), meter),
        graph_store=shadow.graph_store,
    )
    token = active_meter.set(meter)
    try:
        with use_engine(shadow.engine):
            IngestionJob(str(uuid.uuid4()), files, params, seed, ingest).run()
    finally:
        active_meter.reset(token)
    done = bus.done
    if done.get("status") not in ("success", "warning"):
        raise RuntimeError(done.get("error") or f"ingest ended with status {done.get('status')}")
    failed = [d for d in done.get("documents", []) if d.get("error")]
    if failed:
        raise RuntimeError("; ".join(f"{d['filename']}: {d['error']}" for d in failed)[:400])
    with Session(shadow.engine) as s:
        chunks = s.exec(select(func.count()).select_from(Chunk)).one()
    stats = shadow.graph_store.stats()
    with Session(engine) as s:
        live = s.get(EvalIndex, row.id)
        assert live is not None
        live.status, live.finished_at = "ready", _now()
        live.chunk_count, live.vertex_count, live.edge_count = (
            chunks,
            stats["vertices"],
            stats["edges"],
        )
        live.build_tokens = meter.summary()["total"]
        s.add(live)
        s.commit()
    return shadow


def _remove_dir(settings: Settings, row: EvalIndex) -> None:
    _open.pop(row.dir_name, None)
    shutil.rmtree(_paths(settings, row.dir_name), ignore_errors=True)


def delete(key: str, settings: Settings, engine: Engine) -> bool:
    with Session(engine) as s:
        row = s.get(EvalIndex, key)
        if row is None:
            return False
        _remove_dir(settings, row)
        s.delete(row)
        s.commit()
    return True
