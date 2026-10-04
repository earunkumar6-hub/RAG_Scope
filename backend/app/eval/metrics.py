"""Deterministic evaluation metrics (no LLM).

Relevance is decided by text spans, not chunk ids: a golden "relevant chunk" is stored as
(document sha256, start, end) in the parsed text, and a retrieved chunk counts as relevant when
its span overlaps a relevant span by at least ``MIN_OVERLAP`` of the shorter of the two. Ids
embed the document version, so they would silently stop matching after a re-ingest; spans also
let a later step compare indexes built with different chunk sizes.
"""

import re
from typing import Any

import numpy as np

MIN_OVERLAP = 0.5
CITATION_RX = re.compile(r"\s*\[[^\[\]]+\]")

Span = dict[str, Any]  # {"sha256": str, "start": int, "end": int}


def spans_match(a: Span, b: Span, min_overlap: float = MIN_OVERLAP) -> bool:
    if a["sha256"] != b["sha256"]:
        return False
    overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return overlap > 0 and shorter > 0 and overlap / shorter >= min_overlap


def relevance_flags(retrieved: list[list[Span]], relevant: list[Span]) -> list[bool]:
    """One flag per retrieved item. An item may carry several spans (a canonical chunk plus the
    near-duplicates collapsed onto it); any match makes it relevant."""
    return [any(spans_match(s, r) for s in spans for r in relevant) for spans in retrieved]


def hit_rate(flags: list[bool]) -> float:
    return 1.0 if any(flags) else 0.0


def reciprocal_rank(flags: list[bool]) -> float:
    return next((1.0 / i for i, f in enumerate(flags, start=1) if f), 0.0)


def strip_citations(text: str) -> str:
    return CITATION_RX.sub("", text).strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def answer_relevancy(embedder: Any, question: str, answer: str) -> float:  # noqa: ANN401
    """Cosine between the question (query embedding) and the citation-free answer (passage
    embedding), clipped to [0, 1]. Deterministic; no LLM call."""
    vec = embedder.embed_documents([strip_citations(answer)])[0]
    return max(0.0, min(1.0, cosine(embedder.embed_query(question), vec)))


def summarise(rows: list[dict[str, float | None]]) -> dict[str, dict[str, float | int | None]]:
    """Mean of each metric over rows, ignoring missing values; ``n`` = values averaged."""
    names = sorted({k for r in rows for k in r})
    out: dict[str, dict[str, float | int | None]] = {}
    for name in names:
        values = [v for r in rows if (v := r.get(name)) is not None]
        out[name] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "n": len(values),
        }
    return out
