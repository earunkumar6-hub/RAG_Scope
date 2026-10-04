"""S8 knowledge-graph build: LLM entity/relation extraction per unique chunk -> GraphStore.

Each chunk gets one fast-model call returning strict JSON. Output is validated and clipped per
chunk; a chunk whose output is unusable is recorded as a failure and skipped, never failing the
job. Calls run in a small thread pool, but results are upserted in chunk order so the graph is
the same however the calls interleave.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.graph.base import ENTITY_TYPES, EntityIn, GraphStore, RelationIn, normalize_name
from app.llm.base import LLMError, LLMProvider
from app.pipeline.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)

MAX_WORKERS = 4
MAX_ENTITIES = 20
MAX_RELATIONS = 30
MAX_DESCRIPTION = 300
CHUNK_CHARS = 6000  # ~1.5k tokens; chunks are at most 2048 tokens

EXTRACT_SYSTEM = f"""You extract a knowledge graph from one passage of a document.
Reply with a JSON object and nothing else:
{{"entities": [{{"name": str, "type": str, "description": str, "aliases": [str]}}],
  "relations": [{{"source": str, "relation": str, "target": str}}]}}
Rules:
- name: as written in the passage. type: one of {", ".join(ENTITY_TYPES)}.
- description: one short sentence, from the passage only.
- aliases: other names or abbreviations the passage uses for the same entity (may be empty).
- relation: short lowercase verb phrase (e.g. "is part of", "stores", "founded").
- source and target must be names from "entities".
- At most {MAX_ENTITIES} entities and {MAX_RELATIONS} relations; skip trivial or generic ones.
- If the passage has no meaningful entities, return {{"entities": [], "relations": []}}."""


class _Entity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=120)
    type: str = "OTHER"
    description: str = ""
    aliases: list[str] = []


class _Relation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = Field(min_length=1)
    relation: str = Field(min_length=1, max_length=80)
    target: str = Field(min_length=1)


class _Extraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[_Entity] = []
    relations: list[_Relation] = []


def parse_extraction(data: dict, chunk_id: str) -> tuple[list[EntityIn], list[RelationIn]]:
    """Validate one chunk's LLM output; raises ``LLMError`` if it does not fit the schema."""
    try:
        out = _Extraction.model_validate(data)
    except ValidationError as exc:
        raise LLMError(f"extraction does not match schema: {exc.error_count()} error(s)") from exc
    entities: list[EntityIn] = []
    resolve: dict[str, str] = {}  # any surface form in this chunk -> entity key
    for e in out.entities[:MAX_ENTITIES]:
        name = " ".join(e.name.split())
        key = normalize_name(name)
        if not key:
            continue
        type_ = e.type.strip().upper()
        aliases = [a.strip() for a in e.aliases if a.strip() and normalize_name(a) != key][:5]
        entities.append(
            EntityIn(
                name=name,
                type=type_ if type_ in ENTITY_TYPES else "OTHER",
                description=e.description.strip()[:MAX_DESCRIPTION],
                chunk_id=chunk_id,
                aliases=aliases,
            )
        )
        resolve.setdefault(key, key)
        for a in aliases:
            resolve.setdefault(normalize_name(a), key)
    relations = []
    for r in out.relations[:MAX_RELATIONS]:
        src, tgt = resolve.get(normalize_name(r.source)), resolve.get(normalize_name(r.target))
        if src and tgt and src != tgt:
            relations.append(RelationIn(src, r.relation, tgt, chunk_id))
    return entities, relations


def extract_chunk(llm: LLMProvider, chunk_id: str, text: str, seed: int):  # noqa: ANN201
    data = llm.complete_json(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Passage:\n\n{text[:CHUNK_CHARS]}"},
        ],
        temperature=0.0,
        seed=seed,
        # 20 entities with descriptions + 30 relations can pass 1,500 tokens; a cut-off reply is
        # invalid JSON and loses the whole chunk's graph.
        max_tokens=4000,
        fast=True,
    )
    return parse_extraction(data, chunk_id)


@dataclass
class BuildReport:
    chunks_processed: int = 0
    entities: int = 0
    relations: int = 0
    added_vertices: int = 0
    added_edges: int = 0
    merged_vertices: int = 0
    vectors_added: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    new_keys: list[str] = field(default_factory=list)


def build_graph(
    store: GraphStore,
    embedder: Embedder,
    llm: LLMProvider,
    chunks: list[tuple[str, str]],
    seed: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> BuildReport:
    """Extract from ``chunks`` [(chunk_id, text)], upsert into ``store``, embed new entity names."""
    report = BuildReport()
    results: dict[str, tuple[list[EntityIn], list[RelationIn]]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(extract_chunk, llm, cid, text, seed): cid for cid, text in chunks}
        for done, fut in enumerate(as_completed(futures), start=1):
            cid = futures[fut]
            try:
                results[cid] = fut.result()
            except Exception as exc:  # one bad chunk never fails the job
                msg = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
                report.failures.append({"chunk_id": cid, "error": msg[:300]})
            if on_progress:
                on_progress(done, len(chunks))

    for cid, _ in chunks:  # deterministic order
        if cid not in results:
            continue
        entities, relations = results[cid]
        res = store.upsert(entities, relations)
        report.chunks_processed += 1
        report.entities += len(entities)
        report.relations += len(relations)
        report.added_vertices += res.added_vertices
        report.added_edges += res.added_edges
        report.merged_vertices += res.merged_vertices
        report.new_keys.extend(e.key for e in entities)

    missing = store.keys_without_vectors()
    if missing:
        vectors = embedder.embed_documents([name for _, name in missing])
        store.set_vectors(embedder.model_name, [k for k, _ in missing], vectors)
        report.vectors_added = len(missing)
    report.failures.sort(key=lambda f: f["chunk_id"])
    return report
