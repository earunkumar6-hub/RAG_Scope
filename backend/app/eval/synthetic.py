"""Seeded synthetic golden sets: sample unique chunks, ask the LLM for one Q&A pair per chunk.

The same seed over the same corpus picks the same chunks (numpy ``default_rng`` over chunks
sorted by id). Each question's relevant span is its source chunk, so hit rate and MRR apply.
"""

import logging
from collections.abc import Callable

import numpy as np
from sqlmodel import Session, col, select

from app.core.logging import run_id_var
from app.db.models import Chunk, Document, EvalDataset, EvalQuestion
from app.db.session import get_engine
from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)

MIN_CHUNK_TOKENS = 40
MAX_COUNT = 50
SYSTEM = (
    "You write evaluation data for a question-answering system. From the passage, write ONE "
    "specific question that the passage alone fully answers, and its answer in one or two "
    "sentences using only the passage. Do not mention 'the passage' in the question. Reply with "
    'JSON only: {"question": str, "answer": str}.'
)


def sample_chunks(count: int, seed: int) -> list[tuple[Chunk, Document]]:
    """``count`` unique (non-duplicate) chunks of at least ``MIN_CHUNK_TOKENS``, seeded."""
    with Session(get_engine()) as s:
        rows = s.exec(
            select(Chunk, Document)
            .join(Document, col(Document.id) == col(Chunk.document_id))
            .where(col(Chunk.is_duplicate_of).is_(None), Chunk.token_count >= MIN_CHUNK_TOKENS)
            .order_by(col(Chunk.id))
        ).all()
    if not rows:
        return []
    picks = np.random.default_rng(seed).choice(len(rows), size=min(count, len(rows)), replace=False)
    return [tuple(rows[int(i)]) for i in picks]  # type: ignore[misc]


def generate(
    dataset_id: str,
    count: int,
    seed: int,
    llm_factory: Callable[[], LLMProvider],
    mask: Callable[[str], str],
) -> None:
    """Background job: fill ``dataset_id`` with generated questions (``mask`` = PII masking)."""
    run_id_var.set(dataset_id)
    status, error, made, failed = "error", None, 0, 0
    try:
        picks = sample_chunks(count, seed)
        if not picks:
            raise LLMError(f"no unique chunks of at least {MIN_CHUNK_TOKENS} tokens to sample")
        llm = llm_factory()
        for chunk, doc in picks:
            try:
                data = llm.complete_json(
                    [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Passage:\n{chunk.text}"},
                    ],
                    temperature=0.0,
                    seed=seed,
                    max_tokens=400,
                )
                question, answer = str(data.get("question", "")), str(data.get("answer", ""))
                if len(question.strip()) < 3 or not answer.strip():
                    raise LLMError("empty question or answer")
            except LLMError:
                logger.warning("synthetic question failed", extra={"chunk_id": chunk.id})
                failed += 1
                continue
            span = {
                "sha256": doc.sha256,
                "start": chunk.start_offset,
                "end": chunk.end_offset,
                "chunk_id": chunk.id,
                "filename": doc.filename,
                "page": chunk.page,
            }
            with Session(get_engine()) as s:
                s.add(
                    EvalQuestion(
                        dataset_id=dataset_id,
                        idx=made,
                        question=mask(question.strip())[:2000],
                        ground_truth=mask(answer.strip()),
                        relevant_spans=[span],
                    )
                )
                s.commit()
            made += 1
        if made == 0:
            raise LLMError(f"all {failed} generation call(s) failed")
        status = "ready"
    except Exception as exc:
        logger.exception("synthetic generation failed")
        error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        with Session(get_engine()) as s:
            ds = s.get(EvalDataset, dataset_id)
            if ds is not None:
                ds.status, ds.error, ds.question_count = status, error, made
                ds.meta = {**ds.meta, "failed": failed}
                s.add(ds)
                s.commit()
