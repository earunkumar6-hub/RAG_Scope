"""``GET /api/health``: status of each dependency. Never fails; reports ``degraded`` instead."""

import importlib.util
import logging
import time
from typing import Literal

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import Settings
from app.db.session import get_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["health"])

ComponentStatus = Literal["ok", "configured", "not_configured", "unavailable"]
HEALTHY: set[str] = {"ok", "configured"}


class ComponentHealth(BaseModel):
    status: ComponentStatus
    detail: str = ""
    info: dict[str, str] = {}


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, ComponentHealth]


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check_database() -> ComponentHealth:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return ComponentHealth(status="ok", detail="SQLite reachable")
    except Exception as exc:  # health must never raise
        return ComponentHealth(status="unavailable", detail=str(exc))


def check_chroma(settings: Settings, vector_store: object | None) -> ComponentHealth:
    """Uses the app's shared client (Chroma refuses a second client for the same path)."""
    info = {"path": str(settings.chroma_path), "collection": settings.chroma_collection}
    if vector_store is None:
        return ComponentHealth(
            status="unavailable", detail="vector store not initialised", info=info
        )
    try:
        vector_store.heartbeat()
        meta = vector_store.collection_info()
        info |= {"vectors": str(meta["count"]), "embedding_model": str(meta.get("embedding_model"))}
        return ComponentHealth(status="ok", detail="Chroma persistent client ready", info=info)
    except Exception as exc:
        return ComponentHealth(status="unavailable", detail=str(exc), info=info)


def check_graph_store(settings: Settings, graph_store: object | None) -> ComponentHealth:
    """Uses the app's store (stats() is a cheap read that also proves Neo4j connectivity)."""
    info = {"backend": settings.graph_backend}
    if settings.graph_backend == "neo4j":
        info["uri"] = settings.neo4j_uri or ""
    if graph_store is None:
        return ComponentHealth(
            status="unavailable", detail="graph store not initialised", info=info
        )
    try:
        stats = graph_store.stats()
        info |= {"vertices": str(stats["vertices"]), "edges": str(stats["edges"])}
        detail = "Neo4j reachable" if settings.graph_backend == "neo4j" else "NetworkX store loaded"
        return ComponentHealth(status="ok", detail=detail, info=info)
    except Exception as exc:
        return ComponentHealth(status="unavailable", detail=str(exc), info=info)


def check_llm(settings: Settings) -> ComponentHealth:
    """Checks configuration only (no paid API call); Ollama is pinged since it is local."""
    provider = settings.llm_provider
    if provider == "openai":
        info = {"provider": provider, "model": settings.openai_model}
        if settings.openai_api_key is None:
            return ComponentHealth(
                status="not_configured", detail="OPENAI_API_KEY not set", info=info
            )
        return ComponentHealth(status="configured", detail="API key present", info=info)
    if provider == "anthropic":
        info = {"provider": provider, "model": settings.anthropic_model}
        if settings.anthropic_api_key is None:
            return ComponentHealth(
                status="not_configured", detail="ANTHROPIC_API_KEY not set", info=info
            )
        return ComponentHealth(status="configured", detail="API key present", info=info)
    info = {"provider": provider, "model": settings.ollama_model, "url": settings.ollama_base_url}
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        resp.raise_for_status()
        models = {m.get("name") for m in resp.json().get("models", [])}
        if settings.ollama_model not in models:
            return ComponentHealth(
                status="unavailable", detail=f"model {settings.ollama_model} not pulled", info=info
            )
        return ComponentHealth(status="ok", detail="Ollama reachable", info=info)
    except Exception as exc:
        return ComponentHealth(status="unavailable", detail=str(exc), info=info)


def check_embedding(settings: Settings, embedder: object | None) -> ComponentHealth:
    """Checks the backend is usable; the local model itself loads lazily on first use."""
    if settings.embedding_provider == "openai":
        info = {"provider": "openai", "model": settings.openai_embedding_model}
        if settings.openai_api_key is None:
            return ComponentHealth(
                status="not_configured", detail="OPENAI_API_KEY not set", info=info
            )
        return ComponentHealth(status="configured", detail="API key present", info=info)
    info = {"provider": "local", "model": settings.embedding_model}
    if not _installed("sentence_transformers"):
        return ComponentHealth(
            status="unavailable", detail="sentence-transformers not installed", info=info
        )
    if embedder is not None and getattr(embedder, "loaded", False):
        return ComponentHealth(status="ok", detail="model loaded", info=info)
    return ComponentHealth(status="configured", detail="loads on first use", info=info)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report the status of SQLite, Chroma, the graph store, the LLM and the embedding model."""
    settings: Settings = request.app.state.settings
    started = time.perf_counter()
    components = {
        "database": check_database(),
        "chroma": check_chroma(settings, getattr(request.app.state, "vector_store", None)),
        "graph_store": check_graph_store(settings, getattr(request.app.state, "graph_store", None)),
        "llm": check_llm(settings),
        "embedding_model": check_embedding(settings, getattr(request.app.state, "embedder", None)),
    }
    overall = "ok" if all(c.status in HEALTHY for c in components.values()) else "degraded"
    logger.info(
        "health check",
        extra={"status": overall, "ms": round((time.perf_counter() - started) * 1000)},
    )
    return HealthResponse(status=overall, components=components)
