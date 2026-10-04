"""Delete one document from Chroma, SQLite and the knowledge graph.

Near-duplicates elsewhere that pointed at a deleted chunk are re-homed: the smallest-id one is
promoted to canonical, the rest point at it, and the deleted chunk's graph provenance moves to it
(the texts are >= dedup_threshold similar, so its extracted facts still hold). Clusters keep their
ids and labels; sizes shrink and empty clusters are dropped (the next ingest re-clusters).

Order: Chroma, then SQLite, then the kept original upload, then the graph. Every step is
idempotent while the Document row exists, so a failure part-way can be retried; graph provenance
pointing at a missing chunk is ignored by query-time fusion.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlmodel import Session, col, delete, select

from app.db.models import Chunk, Cluster, Document
from app.db.session import get_engine
from app.graph.base import GraphStore
from app.pipeline.ingestion.runner import INGEST_LOCK
from app.pipeline.ingestion.segregator import member_hash
from app.pipeline.ingestion.vector_store import VectorStore


class DocumentNotFound(LookupError):
    pass


class IngestRunning(RuntimeError):
    pass


@dataclass
class DeleteResult:
    document_id: str
    filename: str
    chunks_deleted: int
    vectors_deleted: int
    duplicates_promoted: int
    duplicates_repointed: int
    vertices_removed: int
    edges_removed: int
    clusters_removed: int

    def as_dict(self) -> dict:
        return asdict(self)


def delete_document(
    document_id: str, vs: VectorStore, graph: GraphStore, raw_dir: Path | None = None
) -> DeleteResult:
    """Raises ``DocumentNotFound``, or ``IngestRunning`` if an ingest job holds the lock."""
    if not INGEST_LOCK.acquire(blocking=False):
        raise IngestRunning("an ingest job is running; try again when it finishes")
    try:
        return _delete(document_id, vs, graph, raw_dir)
    finally:
        INGEST_LOCK.release()


def _delete(
    document_id: str, vs: VectorStore, graph: GraphStore, raw_dir: Path | None
) -> DeleteResult:
    with Session(get_engine()) as s:
        doc = s.get(Document, document_id)
        if doc is None:
            raise DocumentNotFound(document_id)
        own = list(s.exec(select(Chunk).where(Chunk.document_id == document_id)).all())
        own_ids = {c.id for c in own}
        orphans = s.exec(
            select(Chunk).where(
                col(Chunk.is_duplicate_of).in_(own_ids), Chunk.document_id != document_id
            )
        ).all()

        # Re-home each deleted canonical's duplicates onto one promoted survivor.
        by_canonical: dict[str, list[Chunk]] = defaultdict(list)
        for o in orphans:
            by_canonical[o.is_duplicate_of].append(o)
        replace: dict[str, str] = {}  # deleted canonical id -> promoted chunk id
        changed: list[Chunk] = []
        for canonical, group in by_canonical.items():
            group.sort(key=lambda c: c.id)
            heir, rest = group[0], group[1:]
            replace[canonical] = heir.id
            heir.is_duplicate_of, heir.duplicate_similarity = None, None
            for c in rest:
                c.is_duplicate_of = heir.id
            changed += group

        # 1. Chroma
        colx = vs.existing_collection()
        vector_ids = vs.ids_for_documents(colx, [document_id]) if colx is not None else []
        if colx is not None:
            vs.update_metadata(
                colx,
                [c.id for c in changed],
                [{"is_duplicate_of": c.is_duplicate_of or ""} for c in changed],
            )
            vs.delete_ids(colx, vector_ids)

        # 2. SQLite
        for c in changed:
            s.add(c)
        promoted_docs = {c.document_id for c in changed if c.is_duplicate_of is None}
        s.exec(delete(Chunk).where(Chunk.document_id == document_id))
        s.delete(doc)
        s.flush()
        for other_id in promoted_docs:
            other = s.get(Document, other_id)
            if other is not None:
                other.duplicate_chunk_count = len(
                    s.exec(
                        select(Chunk.id).where(
                            Chunk.document_id == other_id, col(Chunk.is_duplicate_of).is_not(None)
                        )
                    ).all()
                )
                s.add(other)
        members: dict[int, list[str]] = defaultdict(list)
        for cid, cluster in s.exec(
            select(Chunk.id, Chunk.cluster_id).where(col(Chunk.is_duplicate_of).is_(None))
        ).all():
            members[cluster].append(cid)
        clusters_removed = 0
        for cluster in s.exec(select(Cluster)).all():
            ids = members.get(cluster.id, [])
            if not ids:
                s.delete(cluster)
                clusters_removed += 1
            else:
                cluster.size, cluster.member_hash = len(ids), member_hash(ids)
                s.add(cluster)
        filename, sha256 = doc.filename, doc.sha256
        s.commit()

    if raw_dir is not None:
        (raw_dir / sha256).unlink(missing_ok=True)

    # 3. Knowledge graph
    removed_v, removed_e = graph.remove_chunks(sorted(own_ids), replace)
    graph.flush()

    return DeleteResult(
        document_id=document_id,
        filename=filename,
        chunks_deleted=len(own_ids),
        vectors_deleted=len(vector_ids),
        duplicates_promoted=len(replace),
        duplicates_repointed=len(changed) - len(replace),
        vertices_removed=removed_v,
        edges_removed=removed_e,
        clusters_removed=clusters_removed,
    )
