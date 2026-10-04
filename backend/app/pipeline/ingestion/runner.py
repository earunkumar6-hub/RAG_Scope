"""Ingestion job orchestration: S1 upload -> S7 vector storage, emitting a StageEvent per stage.

Re-ingest rule: a file whose SHA-256 is already ingested with the same chunk_size/chunk_overlap is
skipped (duplicate). With different chunking params the document is re-chunked: a new version is
built, and the previous version's chunks are excluded from dedup/clustering and replaced at S7.
Chunk ids embed the version (``<doc8>-v<version>-<index>``) so an old citation never resolves to
different text.

Jobs run one at a time (global lock): dedup and corpus-wide clustering need a stable corpus.

When ``IngestDeps.raw_dir`` is set, S1 keeps every uploaded file there as ``<sha256>`` (unmasked
original bytes), so eval runs can re-chunk the corpus at other chunk sizes.
"""

import logging
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlmodel import Session, col, delete, select

from app.core.config import Settings
from app.core.events import EventBus
from app.core.logging import run_id_var
from app.db.models import Chunk, Cluster, Document, Run
from app.db.session import get_engine
from app.graph.base import GraphStore
from app.llm.base import LLMError, LLMProvider
from app.pipeline.ingestion.chunker import chunk_text
from app.pipeline.ingestion.embedder import Embedder, pca_2d
from app.pipeline.ingestion.kg_builder import build_graph
from app.pipeline.ingestion.parser import ParsedDocument, count_pages, parse_document
from app.pipeline.ingestion.segregator import (
    cluster_vectors,
    find_duplicates,
    label_clusters,
    member_hash,
)
from app.pipeline.ingestion.tokenizer import get_tokenizer
from app.pipeline.ingestion.upload import MAX_FILE_BYTES, UploadedFile
from app.pipeline.ingestion.vector_store import Neighbour, VectorStore
from app.pipeline.stages import StageRecorder
from app.schemas.events import IngestStageId
from app.schemas.params import IngestParams

logger = logging.getLogger(__name__)

INGEST_STAGES: tuple[IngestStageId, ...] = (
    "S1_upload",
    "S2_parse",
    "S3_tokenize",
    "S4_chunk",
    "S5_embed",
    "S6_segregate",
    "S7_vector_store",
    "S8_kg_build",
)
INGEST_LOCK = threading.Lock()
PREVIEW_CHARS = 1500
SAMPLE_CHUNKS = 30
MAX_POINTS = 2000
MAX_DUPLICATES_SHOWN = 100
HISTOGRAM_BINS = 12


@dataclass
class IngestDeps:
    settings: Settings
    bus: EventBus
    vector_store: VectorStore
    embedder: Embedder
    llm_factory: Callable[[], LLMProvider]
    graph_store: GraphStore
    raw_dir: Path | None = None  # keep original uploads here (None: eval index builds)


def save_raw(raw_dir: Path, sha256: str, content: bytes) -> None:
    path = raw_dir / sha256
    if path.exists():
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


@dataclass
class DocWork:
    """One file moving through the pipeline."""

    file: UploadedFile
    document_id: str
    action: str  # "new" | "rechunk" | "duplicate"
    version: int
    replaces: bool = False  # an earlier version/attempt of this document must be replaced
    page_count: int = 0
    parsed: ParsedDocument | None = None
    total_tokens: int = 0
    failed: str | None = None
    chunks: list[Chunk] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(UTC)


def _preview(text: str, n: int = 240) -> str:
    return text if len(text) <= n else text[:n] + "…"


class IngestionJob:
    def __init__(
        self,
        job_id: str,
        files: list[UploadedFile],
        params: IngestParams,
        seed: int,
        deps: IngestDeps,
    ):
        self.job_id = job_id
        self.files = files
        self.params = params
        self.seed = seed
        self.deps = deps
        self.rec = StageRecorder(deps.bus, job_id)
        self.work: list[DocWork] = []

    # ------------------------------------------------------------------ entry point
    def run(self) -> None:
        """Execute the job; always ends with a terminal ``done`` message and a finished Run row."""
        run_id_var.set(self.job_id)
        status, result = "error", {}
        try:
            for stage_id in INGEST_STAGES:
                self.rec.emit(stage_id, "pending", summary="queued")
            with INGEST_LOCK:
                result = self._pipeline()
            status = result.pop("status")
        except Exception as exc:
            logger.exception("ingestion failed")
            result = {"error": f"{type(exc).__name__}: {exc}"}
            self._mark_failed(result["error"])
        finally:
            with Session(get_engine()) as s:
                run = s.get(Run, self.job_id)
                if run is not None:
                    run.status, run.result, run.finished_at = status, result, _now()
                    s.add(run)
                    s.commit()
            self.deps.bus.close(self.job_id, {"status": status, **result})

    def _active(self) -> list[DocWork]:
        return [w for w in self.work if w.action != "duplicate" and w.failed is None]

    def _skip(self, stage_id: IngestStageId, reason: str) -> None:
        self.rec.emit(stage_id, "success", duration_ms=0, summary=reason, data={"skipped": True})

    def _pipeline(self) -> dict[str, Any]:
        self._s1_upload()
        stages = [
            ("S2_parse", self._s2_parse),
            ("S3_tokenize", self._s3_tokenize),
            ("S4_chunk", self._s4_chunk),
        ]
        for stage_id, fn in stages:
            if not self._active():
                self._skip(stage_id, "nothing to process")
                continue
            fn()
        if not any(w.chunks for w in self._active()):
            for stage_id in ("S5_embed", "S6_segregate", "S7_vector_store", "S8_kg_build"):
                self._skip(stage_id, "nothing to process")
        else:
            vectors = self._s5_embed()
            plan = self._s6_segregate(vectors)
            old_ids = self._s7_store(vectors, plan)
            self._s8_kg_build(old_ids)
        for w in self.work:  # files that failed in S1/S2: flag their newly created rows
            if w.failed and w.action == "new":
                self._set_doc_error(w.document_id, w.failed)
        docs = [
            {
                "document_id": w.document_id,
                "filename": w.file.filename,
                "action": w.action,
                "chunk_count": len(w.chunks),
                "error": w.failed,
            }
            for w in self.work
        ]
        status = "warning" if any(w.failed for w in self.work) else "success"
        return {"status": status, "documents": docs}

    # ------------------------------------------------------------------ S1
    def _s1_upload(self) -> None:
        params_used = {"max_file_mb": MAX_FILE_BYTES // (1024 * 1024), "files": len(self.files)}
        with self.rec.stage("S1_upload", params_used) as st, Session(get_engine()) as s:
            seen_in_batch: dict[str, str] = {}
            for f in self.files:
                sha = f.sha256
                existing = s.exec(select(Document).where(Document.sha256 == sha)).first()
                ready = existing is not None and existing.status == "ready"
                same_params = ready and (
                    existing.chunk_size == self.params.chunk_size
                    and existing.chunk_overlap == self.params.chunk_overlap
                )
                if sha in seen_in_batch:
                    work = DocWork(f, seen_in_batch[sha], "duplicate", 0)
                elif same_params:
                    work = DocWork(f, existing.id, "duplicate", existing.version)
                elif ready:
                    work = DocWork(f, existing.id, "rechunk", existing.version + 1, replaces=True)
                elif existing is not None:  # earlier attempt failed: retry on the same row
                    work = DocWork(f, existing.id, "new", existing.version + 1, replaces=True)
                    existing.filename, existing.extension = f.filename, f.extension
                    existing.chunk_size = self.params.chunk_size
                    existing.chunk_overlap = self.params.chunk_overlap
                    existing.status, existing.error = "processing", None
                    existing.last_job_id, existing.updated_at = self.job_id, _now()
                    s.add(existing)
                else:
                    doc_id = self._new_document_id(s)
                    work = DocWork(f, doc_id, "new", 1)
                    s.add(
                        Document(
                            id=doc_id,
                            filename=f.filename,
                            extension=f.extension,
                            size_bytes=len(f.content),
                            sha256=sha,
                            chunk_size=self.params.chunk_size,
                            chunk_overlap=self.params.chunk_overlap,
                            last_job_id=self.job_id,
                        )
                    )
                seen_in_batch.setdefault(sha, work.document_id)
                if self.deps.raw_dir is not None:
                    save_raw(self.deps.raw_dir, sha, f.content)
                if work.action != "duplicate":
                    try:
                        work.page_count = count_pages(f.extension, f.content)
                    except Exception as exc:  # corrupt file: reported, other files continue
                        work.failed = f"unreadable file: {exc}"
                self.work.append(work)
            s.commit()
            files = [
                {
                    "document_id": w.document_id,
                    "filename": w.file.filename,
                    "size_bytes": len(w.file.content),
                    "sha256": w.file.sha256,
                    "page_count": w.page_count,
                    "action": w.action,
                    "error": w.failed,
                }
                for w in self.work
            ]
            dup = sum(w.action == "duplicate" for w in self.work)
            failed = sum(w.failed is not None for w in self.work)
            st.data = {"files": files}
            st.summary = (
                f"{len(self.work)} file(s): {len(self._active())} to process, {dup} duplicate"
            )
            if failed:
                st.summary += f", {failed} unreadable"
                st.status = "warning"

    @staticmethod
    def _new_document_id(s: Session) -> str:
        while True:  # chunk ids use the first 8 hex chars, keep them unique
            doc_id = uuid.uuid4().hex
            clash = s.exec(select(Document).where(col(Document.id).startswith(doc_id[:8]))).first()
            if clash is None:
                return doc_id

    # ------------------------------------------------------------------ S2
    def _s2_parse(self) -> None:
        with self.rec.stage("S2_parse", {"header_footer_rule": ">50% of pages"}) as st:
            docs = []
            for w in self._active():
                try:
                    w.parsed = parse_document(w.file.extension, w.file.content, PREVIEW_CHARS)
                    if not w.parsed.text.strip():
                        w.failed = "no extractable text (scanned PDF?)"
                except Exception as exc:
                    w.failed = f"parse error: {exc}"
                p = w.parsed
                docs.append(
                    {
                        "document_id": w.document_id,
                        "filename": w.file.filename,
                        "page_count": p.page_count if p else w.page_count,
                        "raw_chars": p.raw_chars if p else 0,
                        "clean_chars": len(p.text) if p else 0,
                        "removed_header_footer_lines": p.removed_lines if p else [],
                        "raw_preview": p.raw_preview if p else "",
                        "clean_preview": p.text[:PREVIEW_CHARS] if p else "",
                        "error": w.failed,
                    }
                )
            ok = [d for d in docs if not d["error"]]
            st.data = {"documents": docs}
            st.summary = (
                f"{len(ok)} document(s) parsed, "
                f"{sum(d['raw_chars'] for d in ok):,} → {sum(d['clean_chars'] for d in ok):,} chars"
            )
            if len(ok) < len(docs):
                st.status = "warning"
                st.summary += f", {len(docs) - len(ok)} failed"

    # ------------------------------------------------------------------ S3
    def _s3_tokenize(self) -> None:
        encoding = self.deps.settings.tokenizer_encoding
        with self.rec.stage("S3_tokenize", {"encoding": encoding}) as st:
            tok = get_tokenizer(encoding)
            docs = []
            for w in self._active():
                w.total_tokens = tok.count(w.parsed.text)
                docs.append(
                    {
                        "document_id": w.document_id,
                        "filename": w.file.filename,
                        "total_tokens": w.total_tokens,
                    }
                )
            first = self._active()[0]
            total = sum(d["total_tokens"] for d in docs)
            st.data = {
                "total_tokens": total,
                "documents": docs,
                "preview": {
                    "document_id": first.document_id,
                    "tokens": tok.preview(first.parsed.text, 300),
                },
            }
            st.summary = f"{total:,} tokens ({encoding})"

    # ------------------------------------------------------------------ S4
    def _s4_chunk(self) -> None:
        size, overlap = self.params.chunk_size, self.params.chunk_overlap
        with self.rec.stage("S4_chunk", {"chunk_size": size, "chunk_overlap": overlap}) as st:
            tok = get_tokenizer(self.deps.settings.tokenizer_encoding)
            per_doc = []
            for w in self._active():
                spans = chunk_text(w.parsed.text, tok, size, overlap)
                prefix = f"{w.document_id[:8]}-v{w.version}"
                w.chunks = [
                    Chunk(
                        id=f"{prefix}-{sp.index}",
                        document_id=w.document_id,
                        chunk_index=sp.index,
                        page=w.parsed.page_at(sp.start_char),
                        page_end=w.parsed.page_at(max(sp.end_char - 1, sp.start_char)),
                        text=sp.text,
                        token_count=sp.token_count,
                        start_offset=sp.start_char,
                        end_offset=sp.end_char,
                        overlap_prev_chars=sp.overlap_prev_chars,
                    )
                    for sp in spans
                ]
                per_doc.append(
                    {
                        "document_id": w.document_id,
                        "filename": w.file.filename,
                        "chunk_count": len(w.chunks),
                    }
                )
            chunks = [c for w in self._active() for c in w.chunks]
            counts = np.array([c.token_count for c in chunks]) if chunks else np.zeros(0)
            hist, edges = np.histogram(counts, bins=HISTOGRAM_BINS, range=(0, size))
            st.data = {
                "chunk_count": len(chunks),
                "documents": per_doc,
                "token_stats": {
                    "min": int(counts.min()) if len(counts) else 0,
                    "mean": round(float(counts.mean()), 1) if len(counts) else 0,
                    "max": int(counts.max()) if len(counts) else 0,
                },
                "histogram": [
                    {"from": int(edges[i]), "to": int(edges[i + 1]), "count": int(hist[i])}
                    for i in range(len(hist))
                ],
                "chunks": [
                    {
                        "chunk_id": c.id,
                        "document_id": c.document_id,
                        "chunk_index": c.chunk_index,
                        "page": c.page,
                        "page_end": c.page_end,
                        "token_count": c.token_count,
                        "start_offset": c.start_offset,
                        "end_offset": c.end_offset,
                        "overlap_prev_chars": c.overlap_prev_chars,
                        "text": c.text,
                    }
                    for c in chunks[:SAMPLE_CHUNKS]
                ],
                "chunks_shown": min(len(chunks), SAMPLE_CHUNKS),
            }
            st.summary = f"{len(chunks)} chunks (≤{size} tokens, {overlap} overlap)"

    # ------------------------------------------------------------------ S5
    def _s5_embed(self) -> np.ndarray:
        emb = self.deps.embedder
        chunks = [c for w in self._active() for c in w.chunks]
        params_used = {"model": emb.model_name, "batch_size": 32, "normalize": "l2"}
        with self.rec.stage("S5_embed", params_used) as st:
            if not emb.loaded:
                self.rec.progress(
                    "S5_embed",
                    f"loading model {emb.model_name} (first use)",
                    {"phase": "loading_model"},
                )
                t0 = time.perf_counter()
                emb.load()
                st.data["model_load_ms"] = round((time.perf_counter() - t0) * 1000)
            dim = emb.dimension
            self._collection = self.deps.vector_store.collection(emb.model_name, dim)
            texts = [c.text for c in chunks]
            truncated, limit = emb.count_truncated(texts)
            last_report = [0.0]

            def on_batch(done: int, total: int) -> None:
                now = time.perf_counter()
                if done == total or now - last_report[0] > 0.5:
                    last_report[0] = now
                    self.rec.progress(
                        "S5_embed",
                        f"embedded {done}/{total}",
                        {"phase": "embedding", "done": done, "total": total},
                    )

            t0 = time.perf_counter()
            vectors = emb.embed_documents(texts, on_batch)
            embed_ms = round((time.perf_counter() - t0) * 1000)
            points = pca_2d(vectors[:MAX_POINTS])
            st.data |= {
                "count": len(chunks),
                "dimension": dim,
                "model": emb.model_name,
                "embed_ms": embed_ms,
                "truncated_count": truncated,
                "max_input_tokens": limit,
                "points": [
                    {"chunk_id": c.id, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4)}
                    for c, p in zip(chunks[:MAX_POINTS], points, strict=True)
                ],
            }
            st.summary = f"{len(chunks)} vectors × {dim} dims in {embed_ms} ms"
            if truncated:
                st.status = "warning"
                st.summary += (
                    f"; {truncated} chunk(s) exceed the model's {limit}-token input and were "
                    "truncated (lower chunk_size to avoid)"
                )
            return vectors

    # ------------------------------------------------------------------ S6
    def _s6_segregate(self, vectors: np.ndarray) -> dict[str, Any]:
        threshold = self.params.dedup_threshold
        params_used = {"dedup_threshold": threshold, "seed": self.seed, "clustering": "kmeans"}
        vs, colx = self.deps.vector_store, self._collection
        chunks = [c for w in self._active() for c in w.chunks]
        ids = [c.id for c in chunks]
        replaced_docs = [w.document_id for w in self._active() if w.replaces]
        with self.rec.stage("S6_segregate", params_used) as st, Session(get_engine()) as s:
            # Existing duplicates whose canonical belongs to a replaced document become canonical.
            orphans: list[Chunk] = []
            if replaced_docs:
                gone = select(Chunk.id).where(col(Chunk.document_id).in_(replaced_docs))
                orphans = list(
                    s.exec(
                        select(Chunk).where(
                            col(Chunk.is_duplicate_of).in_(gone),
                            col(Chunk.document_id).not_in(replaced_docs),
                        )
                    ).all()
                )
            # A neighbour whose canonical is being replaced is itself canonical from now on;
            # resolving through its stale link would point at a chunk S7 deletes.
            orphan_set = {o.id for o in orphans}
            nearest = [
                (
                    Neighbour(nb.chunk_id, nb.similarity, {**nb.metadata, "is_duplicate_of": ""})
                    if nb is not None and nb.chunk_id in orphan_set
                    else nb
                )
                for nb in vs.nearest(colx, vectors, replaced_docs)
            ]
            dups = find_duplicates(ids, vectors, nearest, threshold)
            for c in chunks:
                m = dups.get(c.id)
                if m is not None:
                    c.is_duplicate_of, c.duplicate_similarity = (
                        m.duplicate_of,
                        round(m.similarity, 4),
                    )

            # Corpus-wide clustering over canonical chunks.
            ex_ids, ex_vecs, ex_texts = vs.unique_records(colx, replaced_docs)
            orphan_ids = [o.id for o in orphans]
            if orphan_ids:
                got = colx.get(ids=orphan_ids, include=["embeddings", "documents"])
                ex_ids += list(got["ids"])
                ex_vecs = np.vstack([v for v in (ex_vecs, np.asarray(got["embeddings"])) if v.size])
                ex_texts += list(got["documents"])
            new_unique = [(i, c) for i, c in enumerate(chunks) if c.is_duplicate_of is None]
            all_ids = ex_ids + [c.id for _, c in new_unique]
            parts = [ex_vecs] if ex_vecs.size else []
            if new_unique:
                parts.append(vectors[[i for i, _ in new_unique]])
            texts = dict(zip(ex_ids, ex_texts, strict=True)) | {c.id: c.text for _, c in new_unique}
            assignment = cluster_vectors(all_ids, np.vstack(parts), self.seed) if parts else {}

            members: dict[int, list[str]] = defaultdict(list)
            for cid in sorted(assignment):
                members[assignment[cid]].append(cid)
            hashes = {k: member_hash(v) for k, v in members.items()}
            cached = {
                row.member_hash: (row.label, row.label_source)
                for row in s.exec(select(Cluster)).all()
            }
            labels, fallback_reason = label_clusters(
                {k: [texts[i] for i in v] for k, v in members.items()},
                hashes,
                cached,
                self.deps.llm_factory,
                self.seed,
            )

            # Duplicates inherit their canonical's cluster.
            for c in chunks:
                c.cluster_id = assignment.get(c.is_duplicate_of or c.id, -1)

            existing_text = {}
            dup_targets = [m.duplicate_of for m in dups.values() if m.source == "existing"]
            if dup_targets:
                rows = s.exec(select(Chunk).where(col(Chunk.id).in_(dup_targets))).all()
                existing_text = {r.id: r.text for r in rows}
            batch_text = {c.id: c.text for c in chunks}
            by_id = {c.id: c for c in chunks}
            clusters = [
                {
                    "id": k,
                    "label": labels[k][0],
                    "label_source": labels[k][1],
                    "size": len(members[k]),
                }
                for k in sorted(members)
            ]
            st.data = {
                "unique_count": len(new_unique),
                "duplicate_count": len(dups),
                "promoted_orphans": len(orphan_ids),
                "duplicates": [
                    {
                        "chunk_id": m.chunk_id,
                        "text": _preview(by_id[m.chunk_id].text),
                        "duplicate_of": m.duplicate_of,
                        "duplicate_of_text": _preview(
                            existing_text.get(m.duplicate_of) or batch_text.get(m.duplicate_of, "")
                        ),
                        "similarity": round(m.similarity, 4),
                        "source": m.source,
                    }
                    for m in list(dups.values())[:MAX_DUPLICATES_SHOWN]
                ],
                "clusters": clusters,
                "assignments": {c.id: c.cluster_id for c in chunks[:MAX_POINTS]},
                "label_fallback_reason": fallback_reason,
            }
            st.summary = (
                f"{len(new_unique)} unique, {len(dups)} near-duplicate; "
                f"{len(clusters)} topic cluster(s) over {len(all_ids)} corpus chunks"
            )
            if fallback_reason:
                st.status = "warning"
                st.summary += "; keyword labels (LLM unavailable)"
            return {
                "assignment": assignment,
                "labels": labels,
                "members": members,
                "hashes": hashes,
                "orphans": orphans,
                "replaced_docs": replaced_docs,
            }

    # ------------------------------------------------------------------ S7
    def _s7_store(self, vectors: np.ndarray, plan: dict[str, Any]) -> list[str]:
        """Returns the chunk ids of replaced previous versions (deleted from Chroma)."""
        vs, colx = self.deps.vector_store, self._collection
        assignment: dict[str, int] = plan["assignment"]
        replaced_docs: list[str] = plan["replaced_docs"]
        active = self._active()
        chunks = [c for w in active for c in w.chunks]
        filenames = {w.document_id: w.file.filename for w in active}
        with (
            self.rec.stage("S7_vector_store", {"collection": vs.collection_name}) as st,
            # S8 reads these chunk objects after this session closes: keep them loaded.
            Session(get_engine(), expire_on_commit=False) as s,
        ):
            # Order: upsert new version, commit SQLite, then drop old vectors. Chunk ids are
            # versioned, so old and new coexist briefly; a failure before the commit leaves the
            # previous version fully intact in both stores.
            old_ids = vs.ids_for_documents(colx, replaced_docs)
            vs.upsert(
                colx,
                [c.id for c in chunks],
                vectors,
                [c.text for c in chunks],
                [
                    {
                        "document_id": c.document_id,
                        "filename": filenames[c.document_id],
                        "page": c.page,
                        "page_end": c.page_end,
                        "chunk_index": c.chunk_index,
                        "token_count": c.token_count,
                        "start_offset": c.start_offset,
                        "end_offset": c.end_offset,
                        "cluster_id": c.cluster_id,
                        "is_duplicate_of": c.is_duplicate_of or "",
                    }
                    for c in chunks
                ],
            )

            # SQLite: replace old versions, re-point cluster ids corpus-wide, promote orphans.
            if replaced_docs:
                s.exec(delete(Chunk).where(col(Chunk.document_id).in_(replaced_docs)))
            orphan_ids = {o.id for o in plan["orphans"]}
            new_ids = {c.id for c in chunks}
            changed_ids, changed_meta = [], []
            for row in s.exec(select(Chunk)).all():
                if row.id in new_ids:
                    continue
                promote = row.id in orphan_ids
                canonical = None if promote else row.is_duplicate_of
                cluster = assignment.get(canonical or row.id, row.cluster_id)
                if cluster != row.cluster_id or promote:
                    row.cluster_id, row.is_duplicate_of = cluster, canonical
                    if promote:
                        row.duplicate_similarity = None
                    s.add(row)
                    changed_ids.append(row.id)
                    changed_meta.append({"cluster_id": cluster, "is_duplicate_of": canonical or ""})
            vs.update_metadata(colx, changed_ids, changed_meta)
            for c in chunks:
                s.add(c)

            s.exec(delete(Cluster))
            for k, ids in plan["members"].items():
                label, source = plan["labels"][k]
                s.add(
                    Cluster(
                        id=k,
                        label=label,
                        label_source=source,
                        size=len(ids),
                        member_hash=plan["hashes"][k],
                    )
                )
            for w in active:
                doc = s.get(Document, w.document_id)
                doc.version = w.version
                doc.page_count = w.parsed.page_count
                doc.total_tokens = w.total_tokens
                doc.chunk_count = len(w.chunks)
                doc.duplicate_chunk_count = sum(c.is_duplicate_of is not None for c in w.chunks)
                doc.chunk_size = self.params.chunk_size
                doc.chunk_overlap = self.params.chunk_overlap
                doc.status, doc.error = "ready", None
                doc.last_job_id, doc.updated_at = self.job_id, _now()
                s.add(doc)
            s.commit()
            vs.delete_ids(colx, old_ids)

            total = colx.count()
            st.data = {
                "collection": vs.collection_name,
                "total_vectors": total,
                "inserted": len(chunks),
                "deleted_previous_version": len(old_ids),
                "metadata_updates": len(changed_ids),
                "records": [
                    {
                        "chunk_id": c.id,
                        "document_id": c.document_id,
                        "page": c.page,
                        "chunk_index": c.chunk_index,
                        "cluster_id": c.cluster_id,
                        "is_duplicate_of": c.is_duplicate_of or "",
                        "text": _preview(c.text, 160),
                    }
                    for c in chunks[:SAMPLE_CHUNKS]
                ],
            }
            st.summary = f"upserted {len(chunks)} → collection '{vs.collection_name}' has {total}"
            if old_ids:
                st.summary += f" (replaced {len(old_ids)} from previous version)"
        return old_ids

    # ------------------------------------------------------------------ S8
    def _s8_kg_build(self, replaced_chunk_ids: list[str]) -> None:
        store = self.deps.graph_store
        params = {"build_graph": self.params.build_graph, "graph_backend": store.backend}
        with self.rec.stage("S8_kg_build", params) as st:
            # Replaced chunk versions lose their provenance whether or not we extract again.
            removed_v, removed_e = store.remove_chunks(replaced_chunk_ids)
            todo = [
                (c.id, c.text)
                for w in self._active()
                for c in w.chunks
                if c.is_duplicate_of is None
            ]
            data: dict[str, Any] = {"removed_vertices": removed_v, "removed_edges": removed_e}
            reason = None
            if not self.params.build_graph:
                reason = "build_graph is off"
            elif not todo:
                reason = "no new unique chunks"
            else:
                try:
                    llm = self.deps.llm_factory()
                except LLMError as exc:
                    st.status, reason = "warning", f"LLM not configured ({exc}); graph not built"
            if reason:
                store.flush()
                stats = store.stats()
                st.summary = reason
                st.data = data | {
                    "skipped": st.status != "warning",
                    "vertex_count": stats["vertices"],
                    "edge_count": stats["edges"],
                    "types": stats["types"],
                }
                return

            def on_progress(done: int, total: int) -> None:
                if done == total or done % 10 == 0:
                    self.rec.progress(
                        "S8_kg_build",
                        f"extracted {done}/{total} chunks",
                        {"done": done, "total": total},
                    )

            report = build_graph(store, self.deps.embedder, llm, todo, self.seed, on_progress)
            store.flush()
            stats = store.stats()
            sample = store.neighbourhood(list(dict.fromkeys(report.new_keys)), hops=0, limit=150)
            st.data = data | {
                "chunks_processed": report.chunks_processed,
                "chunks_failed": len(report.failures),
                "failures": report.failures[:20],
                "entities_extracted": report.entities,
                "relations_extracted": report.relations,
                "added_vertices": report.added_vertices,
                "added_edges": report.added_edges,
                "merged_vertices": report.merged_vertices,
                "vectors_added": report.vectors_added,
                "vertex_count": stats["vertices"],
                "edge_count": stats["edges"],
                "types": stats["types"],
                "sample": {
                    "vertices": [v.summary() for v in sample.vertices],
                    "edges": [e.summary() for e in sample.edges],
                },
            }
            st.summary = (
                f"+{report.added_vertices} vertices, +{report.added_edges} edges from "
                f"{report.chunks_processed} chunk(s); graph has {stats['vertices']} vertices, "
                f"{stats['edges']} edges"
            )
            if report.failures:
                st.status = "warning"
                st.summary += f"; {len(report.failures)} chunk(s) failed extraction"

    # ------------------------------------------------------------------ failure handling
    def _set_doc_error(self, document_id: str, error: str) -> None:
        with Session(get_engine()) as s:
            doc = s.get(Document, document_id)
            if doc is not None and doc.status == "processing":
                doc.status, doc.error, doc.updated_at = "error", error[:500], _now()
                s.add(doc)
                s.commit()

    def _mark_failed(self, error: str) -> None:
        for w in self.work:
            if w.action == "new":
                self._set_doc_error(w.document_id, w.failed or error)
