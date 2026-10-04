"""LLM-as-judge metrics (RAGAS-style), shared by online Q10 and offline eval runs.

Each judge is one structured call on the fast model at temperature 0, cached in ``LLMCache`` by
its full input (chunk ids are versioned, so changed text means a new key). Every metric returns
``{"value": float | None, "note": str, ...details}``; ``value`` is None when it does not apply or
the judge failed (``note`` says which).
"""

from typing import Any

from app.eval.metrics import cosine, strip_citations
from app.guardrails.output_guards import FALLBACK, NOT_FOUND, judge_groundedness
from app.llm.base import LLMError, LLMProvider
from app.llm.cache import cached_json

FALLBACK_PREFIX = FALLBACK.split("{")[0]

RELEVANCE_SYSTEM = (
    "You judge context passages retrieved for a question. For each numbered passage decide "
    "whether it contains information useful for answering the question{reference}. Reply with "
    'JSON only: {{"passages": [{{"index": int, "relevant": bool, "reason": str}}]}}, one entry '
    "per passage."
)
RECALL_SYSTEM = (
    "You check how much of a reference answer is covered by context passages. Split the reference "
    "answer into short factual statements. For each, decide whether the passages contain it. "
    'Reply with JSON only: {"statements": [{"text": str, "attributed": bool, "reason": str}]}.'
)
CORRECTNESS_SYSTEM = (
    "You compare an answer with a reference answer to the same question. Ignore citation "
    "markers. 1) Split the reference into short factual statements; for each, covered = true "
    "if the answer conveys it, even in other words. 2) Split the answer into short factual "
    "statements; label each 'supported' if the reference states or implies it, 'contradicted' "
    "if it conflicts with the reference, otherwise 'extra' (detail the reference does not "
    'mention). Reply with JSON only: {"reference": [{"statement": str, "covered": bool}], '
    '"answer": [{"statement": str, "verdict": "supported" | "contradicted" | "extra"}]}.'
)
# Factual F1, semantic similarity (weights as in RAGAS; unlike RAGAS, extra detail is not an fp)
CORRECTNESS_WEIGHTS = (0.75, 0.25)


def _error_note(exc: Exception) -> str:
    msg = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
    return f"judge unavailable ({msg[:150]})"


def _passages(chunks: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(chunks, start=1))


def _call(
    llm: LLMProvider, kind: str, system: str, user: str, seed: int
) -> tuple[dict[str, Any], bool]:
    return cached_json(
        kind,
        llm.fast_model,
        f"{system}\x1e{user}",
        lambda: llm.complete_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            seed=seed,
            max_tokens=1200,
            fast=True,
        ),
    )


def context_precision(
    llm: LLMProvider,
    question: str,
    chunks: list[tuple[str, str]],
    seed: int,
    ground_truth: str | None = None,
) -> dict[str, Any]:
    """Fraction of ``chunks`` (chunk_id, text) judged relevant: to the question alone (online)
    or to the question given its reference answer (offline)."""
    if not chunks:
        return {"value": None, "note": "no context chunks"}
    reference = " given its reference answer" if ground_truth else ""
    user = f"Question: {question}\n\n"
    if ground_truth:
        user += f"Reference answer: {ground_truth}\n\n"
    user += f"Passages:\n{_passages(chunks)}"
    kind = "eval_precision" if ground_truth else "q10_precision"
    try:
        data, cached = _call(llm, kind, RELEVANCE_SYSTEM.format(reference=reference), user, seed)
        rows = {
            int(r["index"]): r
            for r in data.get("passages", [])
            if isinstance(r, dict) and str(r.get("index", "")).isdigit()
        }
        if any(i not in rows for i in range(1, len(chunks) + 1)):
            raise LLMError("judge skipped some passages")
    except Exception as exc:
        return {"value": None, "note": _error_note(exc)}
    verdicts = [
        {
            "chunk_id": cid,
            "relevant": bool(rows[i].get("relevant")),
            "reason": str(rows[i].get("reason", ""))[:300],
        }
        for i, (cid, _) in enumerate(chunks, start=1)
    ]
    value = sum(v["relevant"] for v in verdicts) / len(verdicts)
    return {"value": round(value, 4), "note": "", "verdicts": verdicts, "cached": cached}


def context_recall(
    llm: LLMProvider, ground_truth: str, chunks: list[tuple[str, str]], seed: int
) -> dict[str, Any]:
    """Fraction of reference-answer statements found in the context."""
    if not chunks:
        return {"value": 0.0, "note": "no context chunks", "statements": []}
    user = f"Reference answer: {ground_truth}\n\nPassages:\n{_passages(chunks)}"
    try:
        data, cached = _call(llm, "eval_recall", RECALL_SYSTEM, user, seed)
        rows = [r for r in data.get("statements", []) if isinstance(r, dict)]
        if not rows:
            raise LLMError("judge returned no statements")
    except Exception as exc:
        return {"value": None, "note": _error_note(exc)}
    statements = [
        {
            "text": str(r.get("text", ""))[:300],
            "attributed": bool(r.get("attributed")),
            "reason": str(r.get("reason", ""))[:300],
        }
        for r in rows
    ]
    value = sum(s["attributed"] for s in statements) / len(statements)
    return {"value": round(value, 4), "note": "", "statements": statements, "cached": cached}


def answer_correctness(
    llm: LLMProvider,
    embedder: Any,  # noqa: ANN401
    question: str,
    answer: str,
    ground_truth: str,
    seed: int,
) -> dict[str, Any]:
    """0.75 x factual F1 (judge's tp/fp/fn statements) + 0.25 x answer/reference similarity."""
    user = (
        f"Question: {question}\n\nReference answer: {ground_truth}\n\n"
        f"Answer: {strip_citations(answer)}"
    )
    try:
        data, cached = _call(llm, "eval_correctness", CORRECTNESS_SYSTEM, user, seed)
        # Per-statement verdicts, mapped to tp/fp/fn here: extra detail lands in no group.
        answer_rows = data.get("answer", []) or []
        groups = {
            "tp": [r["statement"] for r in answer_rows if r.get("verdict") == "supported"],
            "fp": [r["statement"] for r in answer_rows if r.get("verdict") == "contradicted"],
            "fn": [r["statement"] for r in data.get("reference", []) or [] if not r.get("covered")],
        }
        groups = {k: [str(s)[:300] for s in v] for k, v in groups.items()}
    except Exception as exc:
        return {"value": None, "note": _error_note(exc)}
    tp, fp, fn = (len(groups[k]) for k in ("tp", "fp", "fn"))
    f1 = tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn else 0.0
    vecs = embedder.embed_documents([strip_citations(answer), ground_truth])
    similarity = max(0.0, cosine(vecs[0], vecs[1]))
    w_f1, w_sim = CORRECTNESS_WEIGHTS
    return {
        "value": round(w_f1 * f1 + w_sim * similarity, 4),
        "note": "",
        "f1": round(f1, 4),
        "similarity": round(similarity, 4),
        **groups,
        "cached": cached,
    }


def faithfulness(
    llm: LLMProvider | None,
    answer: str,
    context_text: str,
    seed: int,
    prior_rows: list[dict[str, Any]] | None = None,
    prior_text: str | None = None,
) -> dict[str, Any]:
    """Fraction of answer sentences the context supports. Reuses Q9's groundedness verdicts
    when they were made on this exact answer; otherwise runs the same judge."""
    if answer == NOT_FOUND or answer.startswith(FALLBACK_PREFIX):
        return {"value": None, "note": "standard not-found/fallback answer: no claims to check"}
    if prior_rows and prior_text == answer:
        rows, source = prior_rows, "q9"
    elif llm is None:
        return {"value": None, "note": "no answer was generated"}
    else:
        try:
            rows, source = judge_groundedness(llm, answer, context_text, seed), "judge"
        except Exception as exc:
            return {"value": None, "note": _error_note(exc)}
    value = sum(r["supported"] for r in rows) / len(rows)
    return {"value": round(value, 4), "note": "", "source": source, "sentences": rows}
