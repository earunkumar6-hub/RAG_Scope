"""S4 token-based recursive chunking.

Windows are measured in tokens: each chunk has at most ``chunk_size`` tokens and consecutive chunks
share exactly ``chunk_overlap`` tokens. Within a window the split point prefers, in order, a
paragraph break, a sentence end, then a plain token boundary - but never makes a chunk shorter than
``max(chunk_size // 2, chunk_overlap + 1)`` tokens (so the window always advances).
"""

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from app.pipeline.ingestion.tokenizer import Tokenizer

PARAGRAPH_RE = re.compile(r"\n\s*\n")
SENTENCE_RE = re.compile(r"[.!?][\"'”’)\]]*(?=\s)")


@dataclass(frozen=True)
class ChunkSpan:
    index: int
    start_token: int
    end_token: int  # exclusive
    start_char: int
    end_char: int  # exclusive
    text: str
    overlap_prev_tokens: int
    overlap_prev_chars: int

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token


def _boundaries(
    text: str, offsets: list[int], pattern: re.Pattern[str], use_start: bool
) -> list[int]:
    """Token indices at which a split is allowed (split happens *before* that token)."""
    out: set[int] = set()
    for m in pattern.finditer(text):
        i = bisect_left(offsets, m.start() if use_start else m.end())
        if 0 < i < len(offsets):
            out.add(i)
    return sorted(out)


def _best_split(candidates: list[int], lo: int, hi: int) -> int | None:
    """Largest candidate in (lo, hi]."""
    j = bisect_right(candidates, hi) - 1
    if j >= 0 and candidates[j] > lo:
        return candidates[j]
    return None


def chunk_text(
    text: str, tokenizer: Tokenizer, chunk_size: int, chunk_overlap: int
) -> list[ChunkSpan]:
    """Split ``text`` into token windows respecting chunk_size / chunk_overlap (in tokens)."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    tokens, offsets = tokenizer.encode_with_offsets(text)
    n = len(tokens)
    if n == 0:
        return []
    char_at = [*offsets, len(text)]  # char_at[i] = start of token i; char_at[n] = end of text
    paragraphs = _boundaries(text, offsets, PARAGRAPH_RE, use_start=True)
    sentences = _boundaries(text, offsets, SENTENCE_RE, use_start=False)
    min_len = max(chunk_size // 2, chunk_overlap + 1)

    spans: list[ChunkSpan] = []
    start = 0
    while True:
        hard_end = min(start + chunk_size, n)
        if hard_end == n:
            end = n
        else:
            lo = start + min_len - 1
            end = (
                _best_split(paragraphs, lo, hard_end)
                or _best_split(sentences, lo, hard_end)
                or hard_end
            )
        prev = spans[-1] if spans else None
        spans.append(
            ChunkSpan(
                index=len(spans),
                start_token=start,
                end_token=end,
                start_char=char_at[start],
                end_char=char_at[end],
                text=text[char_at[start] : char_at[end]],
                overlap_prev_tokens=(prev.end_token - start) if prev else 0,
                overlap_prev_chars=(prev.end_char - char_at[start]) if prev else 0,
            )
        )
        if end == n:
            return spans
        start = end - chunk_overlap
