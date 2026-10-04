"""``POST /api/ingest`` (multipart) and ``GET /api/ingest/{job_id}/events`` (SSE)."""

import json
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel, ValidationError
from sqlmodel import Session

from app.api.sse import run_event_stream
from app.core.events import bus
from app.db.models import Run
from app.db.session import get_engine
from app.eval import runner as eval_runner
from app.pipeline.ingestion.runner import IngestDeps, IngestionJob
from app.pipeline.ingestion.upload import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_BYTES,
    MAX_FILES,
    UploadedFile,
    extension_of,
    sniff_error,
)
from app.schemas.errors import ErrorDetail, JEVError, jev_error_from
from app.schemas.params import IngestOverrides, IngestParams

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


class IngestAccepted(BaseModel):
    job_id: str
    params: IngestParams


def _parse_params(
    raw: str | None, request: Request
) -> tuple[IngestParams | None, list[ErrorDetail]]:
    overrides = IngestOverrides()
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, [ErrorDetail(field="params", message=f"invalid JSON: {exc.msg}")]
        if not isinstance(data, dict):
            return None, [ErrorDetail(field="params", message="must be a JSON object")]
        try:
            overrides = IngestOverrides.model_validate(data)
        except ValidationError as exc:
            return None, jev_error_from(exc, prefix=("params",)).details
    defaults = request.app.state.runtime_config.get().defaults
    try:
        return defaults.resolve_ingest(overrides), []
    except JEVError as exc:
        return None, exc.details


async def _read_files(files: list[UploadFile]) -> tuple[list[UploadedFile], list[ErrorDetail]]:
    errors: list[ErrorDetail] = []
    if len(files) > MAX_FILES:
        return [], [ErrorDetail(field="files", message=f"at most {MAX_FILES} files per request")]
    out: list[UploadedFile] = []
    for i, upload in enumerate(files):
        field = f"files.{i}"
        name = upload.filename or ""
        ext = extension_of(name)
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(
                ErrorDetail(
                    field=field,
                    message=f"'{name}': extension must be one of {', '.join(ALLOWED_EXTENSIONS)}",
                )
            )
            continue
        content = await upload.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            errors.append(ErrorDetail(field=field, message=f"'{name}': larger than 25 MB"))
            continue
        problem = sniff_error(ext, content)
        if problem:
            errors.append(ErrorDetail(field=field, message=f"'{name}': {problem}"))
            continue
        out.append(UploadedFile(filename=name, extension=ext, content=content))
    return out, errors


@router.post("", response_model=IngestAccepted, status_code=202)
async def ingest(
    request: Request,
    background: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description="1-10 files: .pdf .docx .md .txt")],
    params: Annotated[str | None, Form(description="JSON object of IngestParams overrides")] = None,
) -> IngestAccepted:
    """Validate the upload envelope, then start S1–S7 in the background. Returns the job id."""
    if eval_runner.is_running():  # an eval run needs a stable corpus
        raise HTTPException(status_code=409, detail="an eval run is in progress; try again later")
    resolved, param_errors = _parse_params(params, request)
    uploads, file_errors = await _read_files(files)
    if param_errors or file_errors:
        raise JEVError(file_errors + param_errors)

    state = request.app.state
    job_id = str(uuid.uuid4())
    with Session(get_engine()) as s:
        s.add(
            Run(
                id=job_id,
                kind="ingest",
                request={"files": [u.filename for u in uploads], "params": resolved.model_dump()},
            )
        )
        s.commit()
    bus.open(job_id)
    deps = IngestDeps(
        settings=state.settings,
        bus=bus,
        vector_store=state.vector_store,
        embedder=state.embedder,
        llm_factory=state.llm_factory,
        graph_store=state.graph_store,
        raw_dir=state.settings.uploads_dir,
    )
    seed = state.runtime_config.get().defaults.seed
    background.add_task(IngestionJob(job_id, uploads, resolved, seed, deps).run)
    return IngestAccepted(job_id=job_id, params=resolved)


@router.get("/{job_id}/events")
def ingest_events(job_id: str, request: Request) -> Response:
    """SSE stream of StageEvents (S1–S7) followed by ``done``."""
    return run_event_stream(request, bus, job_id, kind="ingest")
