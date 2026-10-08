"""Corpus browsing (``GET /api/documents``, ``/api/chunks``, ``/api/clusters``) and
``DELETE /api/documents/{id}``."""

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlmodel import Session, col, func, select

from app.db.models import Chunk, Cluster, Document
from app.db.session import get_session
from app.eval import runner as eval_runner
from app.guardrails.pii import WITHHELD, get_pii_engine
from app.pipeline.ingestion.deletion import (
    DocumentNotFound,
    IngestRunning,
    delete_document,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])

SessionDep = Annotated[Session, Depends(get_session)]


def chunk_masker(request: Request) -> Callable[[Chunk], Chunk]:
    """Served copy of a chunk with PII masked in its text (when the output pii_leak guard is
    on). The stored chunk, and what retrieval sends to the LLM, are unchanged."""
    if not request.app.state.runtime_config.get().guardrails.output.pii_leak.enabled:
        return lambda chunk: chunk

    def mask(chunk: Chunk) -> Chunk:
        try:
            text = get_pii_engine().mask(chunk.text)[0]
        except Exception:  # never serve what could not be checked
            logger.warning("PII redaction failed; withholding chunk text", exc_info=True)
            text = WITHHELD
        return Chunk.model_validate(chunk.model_dump() | {"text": text})

    return mask


MaskDep = Annotated[Callable[[Chunk], Chunk], Depends(chunk_masker)]


class ChunkPage(BaseModel):
    items: list[Chunk]
    total: int
    offset: int
    limit: int


@router.get("/documents", response_model=list[Document])
def list_documents(session: SessionDep) -> list[Document]:
    """All documents (newest first) with chunk and duplicate counts."""
    return list(session.exec(select(Document).order_by(col(Document.created_at).desc())).all())


@router.get("/documents/{document_id}", response_model=Document)
def get_document(document_id: str, session: SessionDep) -> Document:
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Unknown document '{document_id}'")
    return doc


class DocumentDeleted(BaseModel):
    document_id: str
    filename: str
    chunks_deleted: int
    vectors_deleted: int
    duplicates_promoted: int
    duplicates_repointed: int
    vertices_removed: int
    edges_removed: int
    clusters_removed: int


@router.delete("/documents/{document_id}", response_model=DocumentDeleted)
def remove_document(document_id: str, request: Request) -> DocumentDeleted:
    """Remove a document from Chroma, the knowledge graph, the database and the kept upload (the
    UI confirms first). 409 while an ingest job or an eval run is running."""
    state = request.app.state
    if eval_runner.is_running():
        raise HTTPException(status_code=409, detail="an eval run is in progress; try again later")
    try:
        result = delete_document(
            document_id, state.vector_store, state.graph_store, state.settings.uploads_dir
        )
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Unknown document '{document_id}'") from exc
    except IngestRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return DocumentDeleted(**result.as_dict())


@router.get("/chunks", response_model=ChunkPage)
def list_chunks(
    session: SessionDep,
    mask: MaskDep,
    document_id: str | None = None,
    cluster_id: int | None = None,
    page: Annotated[int | None, Query(ge=1)] = None,
    duplicates: Annotated[bool | None, Query(description="true: only duplicates")] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ChunkPage:
    """Paginated chunks with optional document / cluster / page / duplicate filters."""
    stmt = select(Chunk)
    if document_id is not None:
        stmt = stmt.where(Chunk.document_id == document_id)
    if cluster_id is not None:
        stmt = stmt.where(Chunk.cluster_id == cluster_id)
    if page is not None:
        stmt = stmt.where(Chunk.page <= page, Chunk.page_end >= page)
    if duplicates is not None:
        dup = col(Chunk.is_duplicate_of)
        stmt = stmt.where(dup.is_not(None) if duplicates else dup.is_(None))
    total = session.exec(select(func.count()).select_from(stmt.subquery())).one()
    items = session.exec(
        stmt.order_by(Chunk.document_id, Chunk.chunk_index).offset(offset).limit(limit)
    ).all()
    return ChunkPage(items=[mask(c) for c in items], total=total, offset=offset, limit=limit)


@router.get("/chunks/{chunk_id}", response_model=Chunk)
def get_chunk(chunk_id: str, session: SessionDep, mask: MaskDep) -> Chunk:
    chunk = session.get(Chunk, chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"Unknown chunk '{chunk_id}'")
    return mask(chunk)


@router.get("/clusters", response_model=list[Cluster])
def list_clusters(session: SessionDep) -> list[Cluster]:
    return list(session.exec(select(Cluster).order_by(Cluster.id)).all())
