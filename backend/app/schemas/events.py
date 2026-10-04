"""``StageEvent``: the unit of pipeline observability streamed over SSE."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

IngestStageId = Literal[
    "S1_upload",
    "S2_parse",
    "S3_tokenize",
    "S4_chunk",
    "S5_embed",
    "S6_segregate",
    "S7_vector_store",
    "S8_kg_build",
]
QueryStageId = Literal[
    "Q1_validate",
    "Q2_input_guardrail",
    "Q3_query_embed",
    "Q4_vector_retrieve",
    "Q5_graph_retrieve",
    "Q6_fusion",
    "Q7_rerank",
    "Q8_generate",
    "Q9_output_guardrail",
    "Q10_eval",
]
StageId = IngestStageId | QueryStageId
StageStatus = Literal["pending", "running", "success", "warning", "blocked", "error"]


class StageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    stage_id: StageId
    status: StageStatus
    started_at: datetime
    duration_ms: int | None = Field(default=None, ge=0)
    params_used: dict[str, Any] = {}
    summary: str = ""
    data: dict[str, Any] = {}
