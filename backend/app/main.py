"""FastAPI application factory. Run with ``uvicorn app.main:create_app --factory``."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import config as config_api
from app.api import documents as documents_api
from app.api import eval as eval_api
from app.api import graph as graph_api
from app.api import health as health_api
from app.api import ingest as ingest_api
from app.api import query as query_api
from app.api import runs as runs_api
from app.api import schema as schema_api
from app.core.config import Settings, get_settings
from app.core.events import bus
from app.core.logging import setup_logging
from app.core.runtime_config import RuntimeConfigStore
from app.db.session import init_engine
from app.eval.runner import recover_interrupted
from app.graph.factory import build_graph_store
from app.llm.factory import build_llm
from app.pipeline.ingestion.embedder import build_embedder
from app.pipeline.ingestion.vector_store import VectorStore
from app.pipeline.query.reranker import CrossEncoderReranker
from app.schemas.errors import ErrorDetail, ErrorResponse, JEVError, details_from_errors

logger = logging.getLogger(__name__)


def _error(status: int, body: ErrorResponse) -> JSONResponse:
    return JSONResponse(status_code=status, content=body.model_dump())


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. ``settings`` defaults to the cached environment settings."""
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_engine(settings.sqlite_path)
    recover_interrupted()

    @asynccontextmanager
    async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
        bus.bind_loop(asyncio.get_running_loop())
        logger.info("startup", extra={"env": settings.app_env, "llm": settings.llm_provider})
        yield
        app_.state.graph_store.close()

    app = FastAPI(title="RAGScope API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime_config = RuntimeConfigStore(settings.runtime_config_path)
    app.state.vector_store = VectorStore(settings.chroma_path, settings.chroma_collection)
    app.state.embedder = build_embedder(settings)  # model loads lazily on first use
    app.state.graph_store = build_graph_store(settings)
    app.state.reranker = CrossEncoderReranker(settings.reranker_model)  # loads lazily
    app.state.llm_factory = lambda: build_llm(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = details_from_errors(list(exc.errors()))
        return _error(422, ErrorResponse(error="VALIDATION_ERROR", details=details))

    @app.exception_handler(JEVError)
    async def _jev(_: Request, exc: JEVError) -> JSONResponse:
        return _error(422, ErrorResponse(error="VALIDATION_ERROR", details=exc.details))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 409: "CONFLICT"}.get(
            exc.status_code, "BAD_REQUEST" if exc.status_code < 500 else "INTERNAL_ERROR"
        )
        details = [ErrorDetail(field="", message=str(exc.detail))]
        return _error(exc.status_code, ErrorResponse(error=code, details=details))

    app.include_router(health_api.router)
    app.include_router(schema_api.router)
    app.include_router(config_api.router)
    app.include_router(ingest_api.router)
    app.include_router(documents_api.router)
    app.include_router(query_api.router)
    app.include_router(runs_api.router)
    app.include_router(graph_api.router)
    app.include_router(eval_api.router)
    return app
