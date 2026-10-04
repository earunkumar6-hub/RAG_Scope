"""``GET/PUT /api/config``: runtime defaults and guardrail settings (never secrets)."""

from fastapi import APIRouter, Request

from app.core.runtime_config import RuntimeConfigStore
from app.schemas.config import RuntimeConfig

router = APIRouter(prefix="/api/config", tags=["config"])


def _store(request: Request) -> RuntimeConfigStore:
    return request.app.state.runtime_config


@router.get("", response_model=RuntimeConfig)
def get_config(request: Request) -> RuntimeConfig:
    """Current runtime configuration."""
    return _store(request).get()


@router.put("", response_model=RuntimeConfig)
def put_config(config: RuntimeConfig, request: Request) -> RuntimeConfig:
    """Replace the runtime configuration (validated as a whole)."""
    return _store(request).set(config)
