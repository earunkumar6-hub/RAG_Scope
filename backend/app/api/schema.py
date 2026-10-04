"""``GET /api/schema/{model}``: JSON Schema export for client-side (zod) validation."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.schemas.config import RuntimeConfig
from app.schemas.errors import ErrorResponse
from app.schemas.events import StageEvent
from app.schemas.params import IngestOverrides, QueryOverrides
from app.schemas.query import QueryRequest

router = APIRouter(prefix="/api/schema", tags=["schema"])

EXPORTED_MODELS: dict[str, type[BaseModel]] = {
    "QueryRequest": QueryRequest,
    "QueryParams": QueryOverrides,
    "IngestParams": IngestOverrides,
    "StageEvent": StageEvent,
    "RuntimeConfig": RuntimeConfig,
    "ErrorResponse": ErrorResponse,
}


@router.get("")
def list_schemas() -> list[str]:
    """Names of models whose JSON Schema can be fetched."""
    return sorted(EXPORTED_MODELS)


@router.get("/{model}")
def get_schema(model: str) -> dict[str, Any]:
    """Return the JSON Schema of a whitelisted JEV model."""
    cls = EXPORTED_MODELS.get(model)
    if cls is None:
        raise HTTPException(status_code=404, detail=f"Unknown schema '{model}'")
    return cls.model_json_schema()
