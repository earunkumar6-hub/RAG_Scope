"""``GET /api/graph``: knowledge-graph explorer data (search, entity-type filter, hop expansion)."""

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from app.graph.base import ENTITY_TYPES, GraphStore
from app.schemas.errors import ErrorDetail, JEVError

router = APIRouter(prefix="/api", tags=["graph"])


class GraphResponse(BaseModel):
    vertices: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    seeds: list[str]  # vertex ids matching ``entity`` (empty in overview mode)
    stats: dict[str, Any]  # whole-graph totals: vertices, edges, types


@router.get("/graph", response_model=GraphResponse)
def get_graph(
    request: Request,
    entity: Annotated[str | None, Query(max_length=200, description="Name/alias search")] = None,
    hops: Annotated[int, Query(ge=0, le=3, description="Neighbourhood around matches")] = 1,
    types: Annotated[str | None, Query(description="Comma-separated entity types")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> GraphResponse:
    """Without ``entity``: the highest-degree vertices. With it: the ``hops`` neighbourhood of the
    vertices whose name or alias contains it. ``types`` keeps only vertices of those types (and
    edges between kept vertices)."""
    store: GraphStore = request.app.state.graph_store
    wanted = [t.strip().upper() for t in types.split(",") if t.strip()] if types else []
    unknown = [t for t in wanted if t not in ENTITY_TYPES]
    if unknown:
        raise JEVError(
            [ErrorDetail(field="types", message=f"unknown type(s) {unknown}; use {ENTITY_TYPES}")]
        )
    seeds: list[str] = []
    if entity and entity.strip():
        seeds = store.find(entity, limit=10)
        sub = store.neighbourhood(seeds, hops, limit)
    else:
        sub = store.overview(wanted or None, limit)
    keep = {v.id for v in sub.vertices if not wanted or v.type in wanted}
    return GraphResponse(
        vertices=[
            v.as_dict() | ({"hop": sub.hops[v.id]} if v.id in sub.hops else {})
            for v in sub.vertices
            if v.id in keep
        ],
        edges=[e.as_dict() for e in sub.edges if e.source in keep and e.target in keep],
        seeds=seeds,
        stats=store.stats(),
    )
