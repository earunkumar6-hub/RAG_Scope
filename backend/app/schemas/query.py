"""``QueryRequest`` envelope for ``POST /api/query``."""

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.params import QueryOverrides

STRICT = ConfigDict(extra="forbid")


class QueryFilters(BaseModel):
    model_config = STRICT

    document_ids: list[Annotated[str, StringConstraints(min_length=1, max_length=64)]] | None = (
        Field(default=None, max_length=500)
    )
    cluster_ids: list[Annotated[int, Field(ge=0)]] | None = Field(default=None, max_length=500)


class QueryRequest(BaseModel):
    """Strict query envelope; ``params`` are partial overrides of the runtime defaults."""

    model_config = STRICT

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)]
    collection: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{2,62}$")] = (
        "chunks"
    )
    params: QueryOverrides = QueryOverrides()
    filters: QueryFilters | None = None
    session_id: UUID | None = None
