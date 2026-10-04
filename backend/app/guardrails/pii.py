"""PII detection and masking (Presidio + spaCy ``en_core_web_sm``).

Detected types: email, phone, payment card (Presidio built-ins) plus India's Aadhaar (12 digits,
Verhoeff checksum) and PAN (AAAAA9999A) via custom pattern recognizers. Matches are replaced with
``<TYPE>`` placeholders. The engine loads lazily (first use ~1-2 s) and is shared process-wide.
"""

import os
import threading
from dataclasses import dataclass

PII_TYPES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IN_AADHAAR", "IN_PAN"]
SCORE_THRESHOLD = 0.4  # phone matches score 0.4 (validated by phonenumbers) without context words

# Verhoeff checksum tables (Aadhaar's last digit)
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


@dataclass(frozen=True)
class PiiFinding:
    type: str
    start: int
    end: int


class PiiEngine:
    def __init__(self) -> None:
        self._analyzer = None
        self._lock = threading.Lock()

    def _load(self):  # noqa: ANN202
        with self._lock:
            if self._analyzer is None:
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
                from presidio_analyzer.nlp_engine import NlpEngineProvider

                class AadhaarRecognizer(PatternRecognizer):
                    def validate_result(self, pattern_text: str) -> bool:
                        digits = "".join(ch for ch in pattern_text if ch.isdigit())
                        return len(digits) == 12 and verhoeff_valid(digits)

                nlp = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                    }
                ).create_engine()
                analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
                analyzer.registry.add_recognizer(
                    AadhaarRecognizer(
                        supported_entity="IN_AADHAAR",
                        patterns=[Pattern("aadhaar", r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b", 0.6)],
                        context=["aadhaar", "uid", "uidai"],
                    )
                )
                analyzer.registry.add_recognizer(
                    PatternRecognizer(
                        supported_entity="IN_PAN",
                        # 4th letter encodes the holder type (P person, C company, ...)
                        patterns=[Pattern("pan", r"\b[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]\b", 0.85)],
                        context=["pan", "permanent account"],
                    )
                )
                self._analyzer = analyzer
        return self._analyzer

    @property
    def loaded(self) -> bool:
        return self._analyzer is not None

    def find(self, text: str) -> list[PiiFinding]:
        if not text.strip():
            return []
        results = self._load().analyze(
            text=text, language="en", entities=PII_TYPES, score_threshold=SCORE_THRESHOLD
        )
        # Keep the highest-scoring, longest span where results overlap.
        results.sort(key=lambda r: (-r.score, -(r.end - r.start)))
        kept: list[PiiFinding] = []
        for r in results:
            if all(r.end <= k.start or r.start >= k.end for k in kept):
                kept.append(PiiFinding(r.entity_type, r.start, r.end))
        return sorted(kept, key=lambda f: f.start)

    def mask(self, text: str) -> tuple[str, list[PiiFinding]]:
        """``text`` with every finding replaced by ``<TYPE>``, plus the findings."""
        findings = self.find(text)
        out = text
        for f in reversed(findings):
            out = out[: f.start] + f"<{f.type}>" + out[f.end :]
        return out, findings


_engine = PiiEngine()


def get_pii_engine() -> PiiEngine:
    return _engine


def summarize(findings: list[PiiFinding]) -> dict[str, int]:
    """Counts per type (never the values themselves)."""
    out: dict[str, int] = {}
    for f in findings:
        out[f.type] = out.get(f.type, 0) + 1
    return dict(sorted(out.items()))
