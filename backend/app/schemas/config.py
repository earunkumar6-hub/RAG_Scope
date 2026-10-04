"""Runtime configuration (``GET/PUT /api/config``): pipeline defaults + guardrail settings.

Contains no secrets; provider credentials come only from environment settings.
"""

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.schemas.errors import jev_error_from
from app.schemas.params import (
    ChunkOverlap,
    ChunkSize,
    DedupThreshold,
    GraphHops,
    HybridWeight,
    IngestOverrides,
    IngestParams,
    QueryOverrides,
    QueryParams,
    Seed,
    SimilarityThreshold,
    Temperature,
    TopK,
    TopN,
    overlap_rule,
    top_n_rule,
)

STRICT = ConfigDict(extra="forbid")


class PipelineDefaults(BaseModel):
    """Default values for every tunable pipeline parameter."""

    # validate_default: cross-field rules must also see fields left at their default values.
    model_config = ConfigDict(extra="forbid", validate_default=True)

    chunk_size: ChunkSize = 512
    chunk_overlap: ChunkOverlap = 64
    dedup_threshold: DedupThreshold = 0.95
    top_k: TopK = 10
    top_n: TopN = 4
    similarity_threshold: SimilarityThreshold = 0.35
    temperature: Temperature = 0.2
    seed: Seed = 42
    graph_hops: GraphHops = 1
    hybrid_weight_vector: HybridWeight = 0.7

    check_overlap = field_validator("chunk_overlap")(overlap_rule)
    check_top_n = field_validator("top_n")(top_n_rule)

    def resolve_query(self, overrides: QueryOverrides) -> QueryParams:
        """Merge client overrides onto defaults; re-validates cross-field rules.

        Raises ``JEVError`` with paths under ``params.`` (e.g. client sends only top_k=2 while
        the default top_n is 4).
        """
        merged = {
            name: getattr(self, name) for name in QueryParams.model_fields
        } | overrides.model_dump(exclude_none=True)
        try:
            return QueryParams.model_validate(merged)
        except ValidationError as exc:
            raise jev_error_from(exc, prefix=("params",)) from exc

    def resolve_ingest(self, overrides: IngestOverrides) -> IngestParams:
        """Merge ingest overrides onto defaults; raises ``JEVError`` under ``params.``."""
        merged = {
            name: getattr(self, name) for name in ("chunk_size", "chunk_overlap", "dedup_threshold")
        } | overrides.model_dump(exclude_none=True)
        try:
            return IngestParams.model_validate(merged)
        except ValidationError as exc:
            raise jev_error_from(exc, prefix=("params",)) from exc


class Guard(BaseModel):
    """A guardrail check with an on/off toggle."""

    model_config = STRICT

    enabled: bool = True


class ThresholdGuard(Guard):
    threshold: float = Field(ge=0.0, le=1.0)


class LengthGuard(Guard):
    max_tokens: int = Field(default=512, ge=8, le=4096)


class FormatGuard(Guard):
    max_chars: int = Field(default=6000, ge=100, le=50000)


class GroundednessGuard(Guard):
    max_unsupported_ratio: float = Field(default=0.3, ge=0.0, le=1.0)


class InputGuardrails(BaseModel):
    model_config = STRICT

    length_check: LengthGuard = LengthGuard()
    prompt_injection: ThresholdGuard = ThresholdGuard(threshold=0.8)
    pii_detection: Guard = Guard()
    toxicity: ThresholdGuard = ThresholdGuard(threshold=0.7)
    # bge-small scores unrelated questions up to ~0.50 against topic centroids; on-topic >= ~0.51
    off_topic: ThresholdGuard = ThresholdGuard(threshold=0.5)


class OutputGuardrails(BaseModel):
    model_config = STRICT

    groundedness: GroundednessGuard = GroundednessGuard()
    citation_check: Guard = Guard()
    pii_leak: Guard = Guard()
    no_answer_handling: Guard = Guard()
    format_check: FormatGuard = FormatGuard()


class GuardrailSettings(BaseModel):
    model_config = STRICT

    input: InputGuardrails = InputGuardrails()
    output: OutputGuardrails = OutputGuardrails()


class EvaluationSettings(BaseModel):
    model_config = STRICT

    online: Guard = Guard()  # Q10 per-query metrics (adds one cached LLM judge call)


class RuntimeConfig(BaseModel):
    """Editable configuration persisted to ``data/runtime_config.json``."""

    model_config = STRICT

    defaults: PipelineDefaults = PipelineDefaults()
    guardrails: GuardrailSettings = GuardrailSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
