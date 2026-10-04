"""Pipeline parameter models.

Two flavours per stage group:
* ``*Overrides``: partial, sent by clients; unset fields fall back to runtime defaults.
* ``*Params``: fully resolved values actually used by a run (shown as ``params_used``).

Cross-field rules use ``field_validator`` on the dependent field (declared after the field it
depends on) so errors are reported at the precise path, e.g. ``params.top_n``.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

STRICT = ConfigDict(extra="forbid")

ChunkSize = Annotated[int, Field(ge=128, le=2048, description="Chunk size in tokens")]
ChunkOverlap = Annotated[
    int, Field(ge=0, description="Chunk overlap in tokens; must be <= chunk_size / 2")
]
DedupThreshold = Annotated[float, Field(ge=0.8, le=1.0, description="Near-duplicate cut-off")]
TopK = Annotated[int, Field(ge=1, le=50, description="Candidates fetched from Chroma")]
TopN = Annotated[int, Field(ge=1, le=20, description="Chunks kept after re-ranking; <= top_k")]
SimilarityThreshold = Annotated[float, Field(ge=0.0, le=1.0, description="Cosine cut-off")]
Temperature = Annotated[float, Field(ge=0.0, le=1.0, description="LLM sampling temperature")]
Seed = Annotated[int, Field(ge=0, description="Seed for LLM and any random ops")]
GraphHops = Annotated[int, Field(ge=0, le=3, description="Graph hops; 0 disables KG retrieval")]
HybridWeight = Annotated[
    float, Field(ge=0.0, le=1.0, description="Vector weight in fusion (graph = 1 - value)")
]


def overlap_rule(overlap: int | None, info: ValidationInfo) -> int | None:
    size = info.data.get("chunk_size")
    if overlap is not None and size is not None and overlap > size // 2:
        raise ValueError(f"chunk_overlap must be <= chunk_size / 2 ({size // 2})")
    return overlap


def top_n_rule(top_n: int | None, info: ValidationInfo) -> int | None:
    top_k = info.data.get("top_k")
    if top_n is not None and top_k is not None and top_n > top_k:
        raise ValueError(f"top_n must be <= top_k ({top_k})")
    return top_n


class IngestParams(BaseModel):
    """Resolved ingestion parameters."""

    model_config = STRICT

    chunk_size: ChunkSize
    chunk_overlap: ChunkOverlap
    dedup_threshold: DedupThreshold
    build_graph: bool = True

    check_overlap = field_validator("chunk_overlap")(overlap_rule)


class IngestOverrides(BaseModel):
    """Partial ingestion parameters sent with ``POST /api/ingest`` (JSON form field ``params``)."""

    model_config = STRICT

    chunk_size: ChunkSize | None = None
    chunk_overlap: ChunkOverlap | None = None
    dedup_threshold: DedupThreshold | None = None
    build_graph: bool | None = None

    check_overlap = field_validator("chunk_overlap")(overlap_rule)


class QueryParams(BaseModel):
    """Resolved query parameters."""

    model_config = STRICT

    top_k: TopK
    top_n: TopN
    similarity_threshold: SimilarityThreshold
    temperature: Temperature
    seed: Seed
    graph_hops: GraphHops
    hybrid_weight_vector: HybridWeight

    check_top_n = field_validator("top_n")(top_n_rule)


class QueryOverrides(BaseModel):
    """Partial query parameters sent inside ``QueryRequest.params``."""

    model_config = STRICT

    top_k: TopK | None = None
    top_n: TopN | None = None
    similarity_threshold: SimilarityThreshold | None = None
    temperature: Temperature | None = None
    seed: Seed | None = None
    graph_hops: GraphHops | None = None
    hybrid_weight_vector: HybridWeight | None = None

    check_top_n = field_validator("top_n")(top_n_rule)
