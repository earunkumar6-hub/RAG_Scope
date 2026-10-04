"""Q9 output guardrails, applied to the streamed Q8 answer.

1. citation_check: every cited id must be in the top_n context.
2. groundedness: an LLM judge labels each sentence supported / unsupported by the context;
   unsupported ratio above the limit fails.
   Either failure triggers ONE regeneration (shared), with the problems fed back. If it still
   fails: unknown citations are stripped (repair); ungrounded answers are replaced by a safe
   fallback that points at the most relevant passages.
3. pii_leak: Presidio redaction.
4. no_answer_handling: an answer that says the documents do not cover the question, without any
   citation, is replaced by the standard not-found message.
5. format_check: length limit (cut at a sentence boundary) and balanced markdown.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.guardrails.input_guards import GuardCheck
from app.guardrails.pii import PiiEngine, summarize
from app.llm.base import LLMError, LLMProvider
from app.schemas.config import OutputGuardrails

NOT_FOUND = (
    "I couldn't find relevant information in the ingested documents to answer this question."
)
FALLBACK = (
    "I couldn't produce an answer that is fully supported by the documents. "
    "The most relevant passages are: {citations}."
)
NO_ANSWER_RX = re.compile(
    r"(do(es)? not (cover|contain|mention|include)|don'?t know|not (found|mentioned|covered) in|"
    r"no (relevant )?information|cannot (find|answer)|isn'?t (covered|mentioned))",
    re.I,
)
SENTENCE_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
JUDGE_SYSTEM = (
    "You check whether an answer is supported by its context passages. Split the answer into "
    "sentences. For each sentence decide if the passages support it (a sentence that only says "
    "the documents lack information counts as supported). Reply with JSON only: "
    '{"sentences": [{"text": str, "supported": bool}]}.'
)


@dataclass
class OutputResult:
    answer: str
    checks: list[GuardCheck]
    regenerated: bool = False
    judge: list[dict[str, Any]] = field(default_factory=list)
    judged_text: str | None = None  # the answer text ``judge`` rows refer to


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RX.split(text.strip()) if s.strip()]


def strip_unknown_citations(answer: str, unknown: list[str]) -> str:
    out = answer
    for cid in unknown:
        out = re.sub(rf"\s*\[{re.escape(cid)}\]", "", out)
        out = re.sub(rf",\s*{re.escape(cid)}(?=[\],])|{re.escape(cid)},\s*", "", out)
    return out


def repair_markdown(text: str) -> tuple[str, list[str]]:
    fixes = []
    if text.count("```") % 2:
        text += "\n```"
        fixes.append("closed an unterminated code fence")
    if text.count("**") % 2:
        i = text.rfind("**")
        text = text[:i] + text[i + 2 :]
        fixes.append("removed an unmatched bold marker")
    return text, fixes


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    end = max(cut.rfind(". "), cut.rfind(".\n"))
    return (cut[: end + 1] if end > max_chars // 2 else cut.rstrip()) + " …"


def judge_groundedness(
    llm: LLMProvider, answer: str, context: str, seed: int
) -> list[dict[str, Any]]:
    data = llm.complete_json(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Context passages:\n{context}\n\nAnswer:\n{answer}"},
        ],
        temperature=0.0,
        seed=seed,
        max_tokens=1200,
        fast=True,
    )
    rows = data.get("sentences")
    if not isinstance(rows, list) or not rows:
        raise LLMError("judge returned no sentences")
    return [
        {"text": str(r.get("text", ""))[:500], "supported": bool(r.get("supported"))}
        for r in rows
        if isinstance(r, dict)
    ]


def run_output_guards(
    answer: str,
    messages: list[dict[str, str]],
    context_ids: list[str],
    llm: LLMProvider,
    settings: OutputGuardrails,
    pii: PiiEngine,
    parse_citations: Callable[[str, set[str]], tuple[list[str], list[str]]],
    temperature: float,
    seed: int,
) -> OutputResult:
    valid = set(context_ids)
    context_text = messages[-1]["content"]
    checks: dict[str, GuardCheck] = {}
    regenerated = False
    judged: list[dict[str, Any]] = []
    judged_text: str | None = None

    def citation_problem(text: str) -> list[str]:
        return parse_citations(text, valid)[1]

    def grounding_problem(text: str) -> tuple[float | None, list[str], str]:
        """(unsupported ratio, unsupported sentences, note); ratio None if the judge failed."""
        nonlocal judged, judged_text
        try:
            judged = judge_groundedness(llm, text, context_text, seed)
            judged_text = text
        except Exception as exc:
            msg = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
            return None, [], f"judge unavailable ({msg[:150]})"
        bad = [r["text"] for r in judged if not r["supported"]]
        return len(bad) / len(judged), bad, ""

    cit_on, gr = settings.citation_check.enabled, settings.groundedness
    unknown = citation_problem(answer) if cit_on else []
    ratio, bad, note = grounding_problem(answer) if gr.enabled else (None, [], "")
    failing_grounding = ratio is not None and ratio > gr.max_unsupported_ratio

    if unknown or failing_grounding:
        problems = []
        if unknown:
            problems.append(f"it cited ids that are not in the context: {', '.join(unknown)}")
        if failing_grounding:
            listed = "; ".join(f'"{s}"' for s in bad[:5])
            problems.append(f"these sentences are not supported by the context: {listed}")
        retry = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": "Your answer has problems: "
                + " and ".join(problems)
                + ". Rewrite it using only the context, cite only the chunk ids shown, and say so "
                "if the context does not contain the answer.",
            },
        ]
        try:
            answer = llm.complete(retry, temperature=temperature, seed=seed, max_tokens=1024).text
            regenerated = True
        except Exception as exc:  # a failed retry keeps the first answer; repairs still apply
            msg = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
            note = f"regeneration failed ({msg[:150]})"
        first = {"unknown": unknown, "ratio": ratio}
        unknown = citation_problem(answer) if cit_on else []
        if gr.enabled and regenerated:
            ratio, bad, note = grounding_problem(answer)
            failing_grounding = ratio is not None and ratio > gr.max_unsupported_ratio
    else:
        first = None

    # -- citation_check verdict
    if not cit_on:
        checks["citation_check"] = GuardCheck("citation_check", "skipped", "disabled")
    elif unknown:
        answer = strip_unknown_citations(answer, unknown)
        checks["citation_check"] = GuardCheck(
            "citation_check",
            "repaired",
            f"removed citations not in the context: {', '.join(unknown)}",
            details={"unknown": unknown, "regenerated": regenerated},
        )
    elif first and first["unknown"]:
        checks["citation_check"] = GuardCheck(
            "citation_check",
            "regenerated",
            f"first answer cited ids not in the context ({', '.join(first['unknown'])}); "
            "regenerated answer is clean",
        )
    else:
        checks["citation_check"] = GuardCheck("citation_check", "pass", "all citations in context")

    # -- groundedness verdict
    if not gr.enabled:
        checks["groundedness"] = GuardCheck("groundedness", "skipped", "disabled")
    elif ratio is None:
        checks["groundedness"] = GuardCheck("groundedness", "skipped", note)
    elif failing_grounding:
        cites = " ".join(f"[{cid}]" for cid in context_ids[:3])
        answer = FALLBACK.format(citations=cites)
        checks["groundedness"] = GuardCheck(
            "groundedness",
            "replaced",
            f"{ratio:.0%} of sentences unsupported (limit {gr.max_unsupported_ratio:.0%}) "
            + ("even after regenerating" if regenerated else f"and {note}")
            + "; replaced with a safe fallback",
            score=ratio,
            threshold=gr.max_unsupported_ratio,
            details={"unsupported": bad[:10]},
        )
    elif first and first["ratio"] is not None and first["ratio"] > gr.max_unsupported_ratio:
        checks["groundedness"] = GuardCheck(
            "groundedness",
            "regenerated",
            f"first answer {first['ratio']:.0%} unsupported; regenerated answer {ratio:.0%}",
            score=ratio,
            threshold=gr.max_unsupported_ratio,
        )
    else:
        checks["groundedness"] = GuardCheck(
            "groundedness",
            "pass",
            f"{ratio:.0%} of {len(judged)} sentence(s) unsupported",
            score=ratio,
            threshold=gr.max_unsupported_ratio,
        )

    # -- pii_leak
    if not settings.pii_leak.enabled:
        checks["pii_leak"] = GuardCheck("pii_leak", "skipped", "disabled")
    else:
        redacted, findings = pii.mask(answer)
        if findings:
            answer = redacted
            counts = summarize(findings)
            checks["pii_leak"] = GuardCheck(
                "pii_leak",
                "redacted",
                "redacted " + ", ".join(f"{n} {t.lower()}" for t, n in counts.items()),
                details={"types": counts},
            )
        else:
            checks["pii_leak"] = GuardCheck("pii_leak", "pass", "no PII in the answer")

    # -- no_answer_handling
    if not settings.no_answer_handling.enabled:
        checks["no_answer_handling"] = GuardCheck("no_answer_handling", "skipped", "disabled")
    elif NO_ANSWER_RX.search(answer) and not parse_citations(answer, valid)[0]:
        answer = NOT_FOUND
        checks["no_answer_handling"] = GuardCheck(
            "no_answer_handling", "replaced", "answer says the documents lack it; standardised"
        )
    else:
        checks["no_answer_handling"] = GuardCheck("no_answer_handling", "pass", "answer given")

    # -- format_check
    fmt = settings.format_check
    if not fmt.enabled:
        checks["format_check"] = GuardCheck("format_check", "skipped", "disabled")
    else:
        fixes = []
        if len(answer) > fmt.max_chars:
            answer = truncate(answer, fmt.max_chars)
            fixes.append(f"truncated to {fmt.max_chars} characters")
        answer, md = repair_markdown(answer)
        fixes += md
        checks["format_check"] = GuardCheck(
            "format_check",
            "repaired" if fixes else "pass",
            "; ".join(fixes) if fixes else f"{len(answer)} chars, markdown balanced",
        )

    order = ["citation_check", "groundedness", "pii_leak", "no_answer_handling", "format_check"]
    return OutputResult(answer, [checks[k] for k in order], regenerated, judged, judged_text)
