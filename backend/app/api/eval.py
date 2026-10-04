"""Offline evaluation API: golden sets (JSONL upload or seeded synthetic), eval runs over a grid
of query parameters plus chunk_size, and the throwaway indexes chunk_size cells query. Runs
execute in the background; clients poll ``GET /api/eval/runs/{id}``.
"""

import json
import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from sqlmodel import Session, col, select

from app.core.events import bus
from app.db.models import EvalDataset, EvalIndex, EvalQuestion, EvalResult, EvalRun
from app.db.session import get_engine
from app.eval import indexes, runner, synthetic
from app.guardrails.pii import get_pii_engine
from app.pipeline.ingestion.runner import INGEST_LOCK
from app.pipeline.ingestion.segregator import MAX_CLUSTERS
from app.pipeline.query.runner import QueryDeps
from app.schemas.errors import ErrorDetail, JEVError, jev_error_from
from app.schemas.params import IngestOverrides, IngestParams, QueryOverrides

router = APIRouter(prefix="/api/eval", tags=["eval"])

STRICT = ConfigDict(extra="forbid")
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_GRID_VALUES = 6
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class GoldenLine(BaseModel):
    """One JSONL line of an uploaded golden set."""

    model_config = STRICT

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)
    ]
    ground_truth: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)
    ]
    relevant_chunk_ids: list[str] = Field(default_factory=list, max_length=20)


class SyntheticRequest(BaseModel):
    model_config = STRICT

    name: Name
    count: int = Field(default=10, ge=1, le=synthetic.MAX_COUNT)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class EvalRunRequest(BaseModel):
    model_config = STRICT

    dataset_id: str
    param_grid: dict[str, list[int | float]] = Field(default_factory=dict)


class DatasetOut(BaseModel):
    id: str
    name: str
    source: str
    status: str
    error: str | None
    question_count: int
    meta: dict[str, Any]
    created_at: datetime


class QuestionOut(BaseModel):
    id: int
    idx: int
    question: str
    ground_truth: str
    relevant_spans: list[dict[str, Any]]


class DatasetDetail(DatasetOut):
    questions: list[QuestionOut]


class EvalRunOut(BaseModel):
    id: str
    dataset_id: str
    dataset_name: str
    status: str
    param_grid: dict[str, Any]
    cells: list[dict[str, Any]]
    total: int
    done: int
    summary: list[dict[str, Any]]
    error: str | None
    phase: str | None = None  # e.g. "building the chunk_size=256 index" while running
    created_at: datetime
    finished_at: datetime | None


class IndexBuild(BaseModel):
    chunk_size: int
    cached: bool
    chunks: int  # exact, from chunking the kept uploads
    max_calls: int  # KG extraction (at most one per chunk) + cluster labels; 0 when cached


class EvalEstimate(BaseModel):
    cells: int
    questions: int
    query_calls_max: int
    index_builds: list[IndexBuild]
    total_calls_max: int
    missing_sources: list[str]  # documents to re-upload before chunk_size can vary


class IndexOut(BaseModel):
    id: str
    chunk_size: int
    chunk_overlap: int
    dedup_threshold: float
    build_graph: bool
    embedder: str
    kg_model: str
    status: str
    error: str | None
    chunk_count: int
    vertex_count: int
    edge_count: int
    build_tokens: dict[str, Any]
    stale: bool  # built from a different corpus than the current one
    created_at: datetime
    finished_at: datetime | None


class ResultRow(BaseModel):
    id: int
    cell: int
    question_id: int
    question: str
    ground_truth: str
    status: str
    answer: str
    error: str | None
    metrics: dict[str, Any]
    latency_ms: int
    tokens: dict[str, Any]


class ResultDetail(ResultRow):
    relevant_spans: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    details: dict[str, Any]


class EvalRunDetail(EvalRunOut):
    results: list[ResultRow]


class EvalLimits(BaseModel):
    grid_params: list[str]
    max_cells: int
    max_grid_values: int
    max_questions: int
    max_synthetic: int
    llm_calls_per_question: int  # once per question (cached across cells)
    llm_calls_per_cell: int  # per question, per grid cell
    metrics: list[str]


def _dataset_out(ds: EvalDataset) -> DatasetOut:
    return DatasetOut(**ds.model_dump())


def _mask_fn(request: Request):  # noqa: ANN202
    """PII masking for stored questions, as ``POST /api/query`` does for live queries."""
    if not request.app.state.runtime_config.get().guardrails.input.pii_detection.enabled:
        return lambda text: text
    try:
        engine = get_pii_engine()
    except Exception as exc:  # fail closed rather than store unmasked text
        raise HTTPException(
            status_code=503,
            detail=f"PII detector unavailable ({type(exc).__name__}); disable pii_detection in "
            "Settings to store questions without it",
        ) from exc
    return lambda text: engine.mask(text)[0]


def _query_deps(request: Request) -> QueryDeps:
    state = request.app.state
    return QueryDeps(
        settings=state.settings,
        bus=bus,  # replaced by a CaptureBus per question
        vector_store=state.vector_store,
        embedder=state.embedder,
        reranker=state.reranker,
        llm_factory=state.llm_factory,
        graph_store=state.graph_store,
    )


@router.get("/limits", response_model=EvalLimits)
def limits() -> EvalLimits:
    """Grid and dataset limits, and the worst-case LLM-call counts the Eval page multiplies out
    before a run."""
    return EvalLimits(
        grid_params=list(runner.GRID_PARAMS),
        max_cells=runner.MAX_CELLS,
        max_grid_values=MAX_GRID_VALUES,
        max_questions=runner.MAX_QUESTIONS,
        max_synthetic=synthetic.MAX_COUNT,
        llm_calls_per_question=runner.LLM_CALLS_PER_QUESTION,
        llm_calls_per_cell=runner.LLM_CALLS_PER_CELL,
        metrics=list(runner.METRICS),
    )


# ---------------------------------------------------------------- datasets
@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets() -> list[DatasetOut]:
    with Session(get_engine()) as s:
        rows = s.exec(select(EvalDataset).order_by(col(EvalDataset.created_at).desc())).all()
    return [_dataset_out(d) for d in rows]


@router.get("/datasets/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: str) -> DatasetDetail:
    with Session(get_engine()) as s:
        ds = s.get(EvalDataset, dataset_id)
        if ds is None:
            raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset_id}'")
        questions = s.exec(
            select(EvalQuestion)
            .where(EvalQuestion.dataset_id == dataset_id)
            .order_by(col(EvalQuestion.idx))
        ).all()
    return DatasetDetail(
        **ds.model_dump(), questions=[QuestionOut(**q.model_dump()) for q in questions]
    )


def parse_golden(raw: bytes) -> tuple[list[tuple[int, GoldenLine]], list[ErrorDetail]]:
    """Validate JSONL lines into (line number, line); errors name the line (``line N.field``)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], [ErrorDetail(field="file", message="must be UTF-8 text")]
    lines, errors = [], []
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            lines.append((n, GoldenLine.model_validate(json.loads(line))))
        except json.JSONDecodeError as exc:
            errors.append(ErrorDetail(field=f"line {n}", message=f"invalid JSON: {exc.msg}"))
        except ValidationError as exc:
            errors += jev_error_from(exc, prefix=(f"line {n}",)).details
    if not lines and not errors:
        errors.append(ErrorDetail(field="file", message="no questions found"))
    if len(lines) > runner.MAX_QUESTIONS:
        errors.append(
            ErrorDetail(field="file", message=f"at most {runner.MAX_QUESTIONS} questions")
        )
    return lines, errors


@router.post("/datasets", response_model=DatasetOut, status_code=201)
async def upload_dataset(
    request: Request,
    name: Annotated[str, Form(min_length=1, max_length=100)],
    file: Annotated[
        UploadFile, File(description="JSONL: {question, ground_truth, relevant_chunk_ids?}")
    ],
) -> DatasetOut:
    """Store an uploaded golden set. Relevant chunk ids are resolved to text spans now, so they
    keep matching after documents are re-ingested; unknown ids are rejected."""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise JEVError([ErrorDetail(field="file", message="larger than 2 MB")])
    lines, errors = parse_golden(raw)
    spans = runner.chunk_spans(sorted({cid for _, g in lines for cid in g.relevant_chunk_ids}))
    for n, g in lines:
        unknown = [cid for cid in g.relevant_chunk_ids if cid not in spans]
        if unknown:
            errors.append(
                ErrorDetail(
                    field=f"line {n}.relevant_chunk_ids",
                    message=f"unknown chunk id(s): {', '.join(unknown)}",
                )
            )
    if errors:
        raise JEVError(errors)
    mask = _mask_fn(request)
    ds = EvalDataset(
        id=str(uuid.uuid4()),
        name=name.strip(),
        source="upload",
        question_count=len(lines),
        meta={"filename": file.filename},
    )
    with Session(get_engine()) as s:
        s.add(ds)
        s.flush()  # the dataset row must exist before its questions reference it
        for i, (_, g) in enumerate(lines):
            s.add(
                EvalQuestion(
                    dataset_id=ds.id,
                    idx=i,
                    question=mask(g.question),
                    ground_truth=mask(g.ground_truth),
                    relevant_spans=[
                        {k: v for k, v in spans[cid].items() if k != "text"} | {"chunk_id": cid}
                        for cid in g.relevant_chunk_ids
                    ],
                )
            )
        s.commit()
        s.refresh(ds)
        return _dataset_out(ds)


@router.post("/datasets/synthetic", response_model=DatasetOut, status_code=202)
def synthetic_dataset(
    body: SyntheticRequest, request: Request, background: BackgroundTasks
) -> DatasetOut:
    """Start generating a seeded synthetic golden set from the ingested chunks."""
    mask = _mask_fn(request)
    ds = EvalDataset(
        id=str(uuid.uuid4()),
        name=body.name,
        source="synthetic",
        status="generating",
        meta={"count": body.count, "seed": body.seed},
    )
    with Session(get_engine()) as s:
        s.add(ds)
        s.commit()
        s.refresh(ds)
        out = _dataset_out(ds)
    background.add_task(
        synthetic.generate, ds.id, body.count, body.seed, request.app.state.llm_factory, mask
    )
    return out


# ---------------------------------------------------------------- runs
def resolve_grid(grid: dict[str, list[int | float]], request: Request) -> list[dict[str, Any]]:
    """Validate the grid and return every cell's resolved QueryParams; with chunk_size in the
    grid, a cell also carries its resolved IngestParams (other ingest values from the defaults).
    """
    errors = []
    for key, values in grid.items():
        if key in runner.NOT_IN_GRID:
            errors.append(
                ErrorDetail(
                    field=f"param_grid.{key}",
                    message="only chunk_size can vary among ingest parameters; the others use "
                    "the defaults",
                )
            )
        elif key not in runner.GRID_PARAMS:
            errors.append(ErrorDetail(field=f"param_grid.{key}", message="unknown parameter"))
        elif not 1 <= len(values) <= MAX_GRID_VALUES or len(set(values)) != len(values):
            errors.append(
                ErrorDetail(
                    field=f"param_grid.{key}",
                    message=f"1-{MAX_GRID_VALUES} distinct values",
                )
            )
    if errors:
        raise JEVError(errors)
    cells = runner.expand_grid(grid)
    if len(cells) > runner.MAX_CELLS:
        raise JEVError(
            [
                ErrorDetail(
                    field="param_grid",
                    message=f"{len(cells)} combinations; at most {runner.MAX_CELLS}",
                )
            ]
        )
    defaults = request.app.state.runtime_config.get().defaults
    resolved = []
    for i, cell in enumerate(cells, start=1):
        query = {k: v for k, v in cell.items() if k != "chunk_size"}
        try:
            values = defaults.resolve_query(QueryOverrides.model_validate(query)).model_dump()
            if "chunk_size" in cell:
                ingest = IngestOverrides.model_validate({"chunk_size": cell["chunk_size"]})
                values |= defaults.resolve_ingest(ingest).model_dump()
            resolved.append(values)
        except ValidationError as exc:
            errors += jev_error_from(exc, prefix=("param_grid", f"cell {i}")).details
        except JEVError as exc:
            errors += [
                ErrorDetail(
                    field=f"param_grid.cell {i}.{d.field.removeprefix('params.')}",
                    message=d.message,
                )
                for d in exc.details
            ]
    if errors:
        unique = {(d.field, d.message): d for d in errors}  # one chunk_size error per value
        raise JEVError(list(unique.values()))
    return resolved


def _plan(body: EvalRunRequest, request: Request) -> tuple[list[dict[str, Any]], EvalDataset]:
    cells = resolve_grid(body.param_grid, request)
    with Session(get_engine()) as s:
        ds = s.get(EvalDataset, body.dataset_id)
        if ds is None:
            raise JEVError([ErrorDetail(field="dataset_id", message="unknown dataset")])
        if ds.status != "ready" or ds.question_count == 0:
            raise JEVError([ErrorDetail(field="dataset_id", message=f"dataset is {ds.status}")])
    return cells, ds


def _index_params(cells: list[dict[str, Any]]) -> dict[int, IngestParams]:
    return {
        c["chunk_size"]: IngestParams.model_validate({k: c[k] for k in runner.INGEST_KEYS})
        for c in cells
        if "chunk_size" in c
    }


@router.post("/estimate", response_model=EvalEstimate)
def estimate(body: EvalRunRequest, request: Request) -> EvalEstimate:
    """Worst-case LLM calls for a run, including the throwaway indexes it would build."""
    cells, ds = _plan(body, request)
    state = request.app.state
    seed = state.runtime_config.get().defaults.seed
    deps = _query_deps(request)
    builds, missing = [], []
    by_size = _index_params(cells)
    if by_size:
        model = indexes.kg_model(deps)
        with Session(get_engine()) as s:
            docs = indexes.corpus(s)
            missing = indexes.missing_sources(docs, state.settings)
            for size, params in sorted(by_size.items()):
                key = indexes.index_key(
                    [d.sha256 for d in docs], params, seed, deps.embedder.model_name, model
                )
                row = s.get(EvalIndex, key)
                cached = row is not None and row.status == "ready"
                chunks = indexes.count_chunks(docs, state.settings, size, params.chunk_overlap)
                calls = 0 if cached else (chunks if params.build_graph else 0) + MAX_CLUSTERS
                builds.append(
                    IndexBuild(chunk_size=size, cached=cached, chunks=chunks, max_calls=calls)
                )
    q = ds.question_count
    query_calls = q * (runner.LLM_CALLS_PER_QUESTION + runner.LLM_CALLS_PER_CELL * len(cells))
    return EvalEstimate(
        cells=len(cells),
        questions=q,
        query_calls_max=query_calls,
        index_builds=builds,
        total_calls_max=query_calls + sum(b.max_calls for b in builds),
        missing_sources=missing,
    )


def _phase(run: EvalRun, s: Session) -> str | None:
    if run.status != "running":
        return None
    building = s.exec(select(EvalIndex).where(EvalIndex.status == "building")).first()
    return f"building the chunk_size={building.chunk_size} index" if building else None


def _run_out(run: EvalRun, dataset_name: str, phase: str | None = None) -> dict[str, Any]:
    return {**run.model_dump(), "dataset_name": dataset_name, "phase": phase}


@router.get("/runs", response_model=list[EvalRunOut])
def list_runs() -> list[EvalRunOut]:
    with Session(get_engine()) as s:
        rows = s.exec(
            select(EvalRun, EvalDataset)
            .join(EvalDataset, col(EvalDataset.id) == col(EvalRun.dataset_id))
            .order_by(col(EvalRun.created_at).desc())
        ).all()
    return [EvalRunOut(**_run_out(r, d.name)) for r, d in rows]


@router.post("/runs", status_code=202)
def start_run(
    body: EvalRunRequest, request: Request, background: BackgroundTasks
) -> dict[str, str]:
    """Evaluate a ready dataset over every cell of ``param_grid`` (query-time parameters, plus
    chunk_size: those cells query throwaway indexes built from the kept original uploads)."""
    cells, ds = _plan(body, request)
    state = request.app.state
    if state.vector_store.count() == 0:
        raise HTTPException(
            status_code=409, detail="the collection is empty; ingest documents first"
        )
    if INGEST_LOCK.locked():
        raise HTTPException(status_code=409, detail="an ingest job is running; try again later")
    if _index_params(cells):
        with Session(get_engine()) as s:
            missing = indexes.missing_sources(indexes.corpus(s), state.settings)
        if missing:
            raise HTTPException(
                status_code=409,
                detail="original uploads were not kept for: "
                + ", ".join(missing)
                + ". Upload these files again (same settings: they are only stored, not "
                "re-ingested) to vary chunk_size.",
            )
    run_id = str(uuid.uuid4())
    if not runner.try_start(run_id):
        raise HTTPException(status_code=409, detail="another eval run is in progress")
    config = state.runtime_config.get()
    try:
        with Session(get_engine()) as s:
            s.add(
                EvalRun(
                    id=run_id,
                    dataset_id=ds.id,
                    param_grid=body.param_grid,
                    cells=cells,
                    total=len(cells) * ds.question_count,
                )
            )
            s.commit()
    except Exception:
        runner.release(run_id)
        raise
    job = runner.EvalJob(
        run_id,
        _query_deps(request),
        config.guardrails,
        state.vector_store.collection_name,
        config.defaults.seed,
    )
    background.add_task(job.run)
    return {"eval_run_id": run_id}


@router.get("/runs/{run_id}", response_model=EvalRunDetail)
def get_run(run_id: str, cell: int = 0) -> EvalRunDetail:
    """Run status, per-cell metric means, and the per-question results of grid cell ``cell``."""
    with Session(get_engine()) as s:
        run = s.get(EvalRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown eval run '{run_id}'")
        ds = s.get(EvalDataset, run.dataset_id)
        phase = _phase(run, s)
        rows = s.exec(
            select(EvalResult, EvalQuestion)
            .join(EvalQuestion, col(EvalQuestion.id) == col(EvalResult.question_id))
            .where(EvalResult.eval_run_id == run_id, EvalResult.cell == cell)
            .order_by(col(EvalQuestion.idx))
        ).all()
    results = [
        ResultRow(
            **r.model_dump(exclude={"retrieved", "details"}),
            question=q.question,
            ground_truth=q.ground_truth,
        )
        for r, q in rows
    ]
    return EvalRunDetail(**_run_out(run, ds.name if ds else "", phase), results=results)


@router.get("/results/{result_id}", response_model=ResultDetail)
def get_result(result_id: int) -> ResultDetail:
    """One question's drill-down: retrieved chunks and the judges' reasoning."""
    with Session(get_engine()) as s:
        r = s.get(EvalResult, result_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"Unknown eval result {result_id}")
        q = s.get(EvalQuestion, r.question_id)
        assert q is not None
    return ResultDetail(
        **r.model_dump(),
        question=q.question,
        ground_truth=q.ground_truth,
        relevant_spans=q.relevant_spans,
    )


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, bool]:
    """Stop a running eval after its current question (results so far are kept)."""
    with Session(get_engine()) as s:
        if s.get(EvalRun, run_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown eval run '{run_id}'")
    return {"cancelled": runner.request_cancel(run_id)}


# ---------------------------------------------------------------- throwaway indexes
@router.get("/indexes", response_model=list[IndexOut])
def list_indexes() -> list[IndexOut]:
    """Throwaway per-chunk_size indexes; ``stale`` ones were built from an older corpus."""
    with Session(get_engine()) as s:
        current = sorted(d.sha256 for d in indexes.corpus(s))
        rows = s.exec(select(EvalIndex).order_by(col(EvalIndex.created_at).desc())).all()
    return [
        IndexOut(**r.model_dump(exclude={"corpus", "dir_name"}), stale=sorted(r.corpus) != current)
        for r in rows
    ]


@router.delete("/indexes/{index_id}")
def delete_index(index_id: str, request: Request) -> dict[str, bool]:
    """Delete a throwaway index and its files (409 while an eval run is in progress)."""
    if runner.is_running():
        raise HTTPException(status_code=409, detail="an eval run is in progress")
    if not indexes.delete(index_id, request.app.state.settings, get_engine()):
        raise HTTPException(status_code=404, detail=f"Unknown index '{index_id}'")
    return {"deleted": True}
