"""Neo4j ``GraphStore`` (used when ``NEO4J_URI`` is set; docker-compose runs Neo4j 5).

Model: ``(:Entity {key, name, type, description, aliases, alias_keys, alias_norms, chunk_ids,
vector})`` and ``(:Entity)-[:REL {relation, chunk_ids}]->(:Entity)``; ``(:GraphMeta)`` records
the embedding model of the name vectors. Name vectors use a Neo4j vector index (cosine). Neo4j
reports cosine scores as (1 + cos) / 2; they are converted back so thresholds mean the same in
every store. ``namespace`` suffixes the labels/index so tests never touch real data.

Schema (constraint, vector index) is created lazily on first use, so an unreachable Neo4j does
not stop the API from starting; the health check reports it instead.
"""

import re
import threading
from collections import Counter, deque
from typing import Any

import numpy as np

from app.graph.base import (
    Edge,
    EntityIn,
    GraphMismatchError,
    GraphStore,
    RelationIn,
    Subgraph,
    UpsertResult,
    Vertex,
    normalize_name,
    query_ngrams,
)
from app.graph.networkx_store import normalize_relation


class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, namespace: str = ""):
        from neo4j import GraphDatabase

        if namespace and not re.fullmatch(r"[A-Za-z0-9_]+", namespace):
            raise ValueError("namespace must be alphanumeric/underscore")
        suffix = f"_{namespace}" if namespace else ""
        self.L = f"Entity{suffix}"  # node label (identifiers cannot be query parameters)
        self.R = f"REL{suffix}"
        self.M = f"GraphMeta{suffix}"
        self.index = f"entity_name_vectors{suffix}"
        self._driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5.0)
        self._schema_ready = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing
    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self._ensure_schema()
        records, _, _ = self._driver.execute_query(query, params, database_="neo4j")
        return [r.data() for r in records]

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._lock:
            if not self._schema_ready:
                self._driver.execute_query(
                    f"CREATE CONSTRAINT {self.L.lower()}_key IF NOT EXISTS "
                    f"FOR (e:{self.L}) REQUIRE e.key IS UNIQUE",
                    database_="neo4j",
                )
                self._schema_ready = True

    def _vector_index_exists(self) -> bool:
        rows = self._run("SHOW INDEXES YIELD name WHERE name = $n RETURN name", n=self.index)
        return bool(rows)

    def close(self) -> None:
        self._driver.close()

    def wipe(self) -> None:
        """Delete everything in this namespace (tests)."""
        self._run(f"MATCH (e:{self.L}) DETACH DELETE e")
        self._run(f"MATCH (m:{self.M}) DELETE m")
        self._run(f"DROP INDEX {self.index} IF EXISTS")

    # ------------------------------------------------------------------ writes
    def _resolve(self, tx, key: str) -> str | None:  # noqa: ANN001
        rec = tx.run(f"MATCH (e:{self.L} {{key: $k}}) RETURN e.key AS k", k=key).single()
        if rec:
            return rec["k"]
        rec = tx.run(
            f"MATCH (e:{self.L}) WHERE $k IN e.alias_keys RETURN e.key AS k ORDER BY k LIMIT 1",
            k=key,
        ).single()
        return rec["k"] if rec else None

    def upsert(self, entities: list[EntityIn], relations: list[RelationIn]) -> UpsertResult:
        self._ensure_schema()
        counts = {"v": 0, "e": 0, "m": 0}

        def work(tx) -> None:  # noqa: ANN001
            for e in entities:
                key = e.key
                if not key:
                    continue
                target = self._resolve(tx, key) or key
                exists = tx.run(f"MATCH (n:{self.L} {{key: $k}}) RETURN n", k=target).single()
                if exists:
                    tx.run(
                        f"""MATCH (n:{self.L} {{key: $k}})
                        SET n.chunk_ids = CASE WHEN $c IN n.chunk_ids THEN n.chunk_ids
                                               ELSE n.chunk_ids + $c END,
                            n.description = CASE WHEN n.description = '' THEN $d
                                                 ELSE n.description END,
                            n.type = CASE WHEN n.type = 'OTHER' AND $t <> 'OTHER' THEN $t
                                          ELSE n.type END,
                            n.aliases = CASE WHEN $name = n.name OR $name IN n.aliases
                                             THEN n.aliases ELSE n.aliases + $name END,
                            n.alias_norms = CASE WHEN $name = n.name OR $nk IN n.alias_norms
                                                 THEN n.alias_norms ELSE n.alias_norms + $nk END""",
                        k=target,
                        c=e.chunk_id,
                        d=e.description,
                        t=e.type,
                        name=e.name,
                        nk=normalize_name(e.name),
                    )
                    counts["m"] += 1
                else:
                    tx.run(
                        f"""CREATE (:{self.L} {{key: $k, name: $name, type: $t, description: $d,
                            aliases: [], alias_keys: [], alias_norms: [], chunk_ids: [$c]}})""",
                        k=target,
                        name=e.name,
                        t=e.type,
                        d=e.description,
                        c=e.chunk_id,
                    )
                    counts["v"] += 1
                for alias in e.aliases:
                    ak = normalize_name(alias)
                    if not ak or ak == target:
                        continue
                    taken = tx.run(
                        f"MATCH (n:{self.L}) WHERE n.key = $a OR $a IN n.alias_keys "
                        "RETURN n LIMIT 1",
                        a=ak,
                    ).single()
                    if not taken:
                        tx.run(
                            f"""MATCH (n:{self.L} {{key: $k}})
                            SET n.alias_keys = n.alias_keys + $a,
                                n.aliases = CASE WHEN $alias IN n.aliases THEN n.aliases
                                                 ELSE n.aliases + $alias END,
                                n.alias_norms = CASE WHEN $a IN n.alias_norms THEN n.alias_norms
                                                     ELSE n.alias_norms + $a END""",
                            k=target,
                            a=ak,
                            alias=alias,
                        )
            for r in relations:
                src, tgt = self._resolve(tx, r.source), self._resolve(tx, r.target)
                rel = normalize_relation(r.relation)
                if src is None or tgt is None or src == tgt or not rel:
                    continue
                found = tx.run(
                    f"""MATCH (:{self.L} {{key: $s}})-[x:{self.R} {{relation: $r}}]->
                          (:{self.L} {{key: $t}})
                    SET x.chunk_ids = CASE WHEN $c IN x.chunk_ids THEN x.chunk_ids
                                           ELSE x.chunk_ids + $c END
                    RETURN count(x) AS n""",
                    s=src,
                    t=tgt,
                    r=rel,
                    c=r.chunk_id,
                ).single()["n"]
                if not found:
                    tx.run(
                        f"""MATCH (s:{self.L} {{key: $s}}), (t:{self.L} {{key: $t}})
                        CREATE (s)-[:{self.R} {{relation: $r, chunk_ids: [$c]}}]->(t)""",
                        s=src,
                        t=tgt,
                        r=rel,
                        c=r.chunk_id,
                    )
                    counts["e"] += 1

        with self._driver.session(database="neo4j") as session:
            session.execute_write(work)
        return UpsertResult(counts["v"], counts["e"], counts["m"])

    def keys_without_vectors(self) -> list[tuple[str, str]]:
        rows = self._run(
            f"MATCH (e:{self.L}) WHERE e.vector IS NULL RETURN e.key AS k, e.name AS n ORDER BY k"
        )
        return [(r["k"], r["n"]) for r in rows]

    def set_vectors(self, model: str, keys: list[str], vectors: np.ndarray) -> None:
        if not keys:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        meta = self._run(f"MATCH (m:{self.M}) RETURN m.vector_model AS model")
        if meta and meta[0]["model"] and meta[0]["model"] != model:
            raise GraphMismatchError(
                f"Graph entity vectors were built with {meta[0]['model']}, not {model}"
            )
        self._run(f"MERGE (m:{self.M} {{id: 'meta'}}) SET m.vector_model = $model", model=model)
        if not self._vector_index_exists():
            self._run(
                f"CREATE VECTOR INDEX {self.index} IF NOT EXISTS FOR (e:{self.L}) ON (e.vector) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(vectors.shape[1])}, "
                "`vector.similarity_function`: 'cosine'}}"
            )
            self._run("CALL db.awaitIndexes(60)")
        self._run(
            f"""UNWIND $rows AS row MATCH (e:{self.L} {{key: row.k}})
            WHERE e.vector IS NULL SET e.vector = row.v""",
            rows=[{"k": k, "v": v.tolist()} for k, v in zip(keys, vectors, strict=True)],
        )

    def remove_chunks(
        self, chunk_ids: list[str], replace: dict[str, str] | None = None
    ) -> tuple[int, int]:
        replace = replace or {}
        ids = sorted(set(chunk_ids) | set(replace))
        if not ids:
            return 0, 0
        # Kept ids, plus each replaced id's substitute, de-duplicated.
        new_ids = (
            "reduce(acc = [], c IN [c IN {v}.chunk_ids WHERE NOT c IN $ids] + "
            "[c IN {v}.chunk_ids WHERE c IN keys($rep) | $rep[c]] | "
            "CASE WHEN c IN acc THEN acc ELSE acc + c END)"
        )
        edges = self._run(
            f"""MATCH (:{self.L})-[r:{self.R}]->(:{self.L})
            WHERE any(c IN r.chunk_ids WHERE c IN $ids)
            SET r.chunk_ids = {new_ids.format(v="r")}
            WITH r WHERE size(r.chunk_ids) = 0
            DELETE r RETURN count(*) AS n""",
            ids=ids,
            rep=replace,
        )[0]["n"]
        dead = self._run(
            f"""MATCH (e:{self.L}) WHERE any(c IN e.chunk_ids WHERE c IN $ids)
            SET e.chunk_ids = {new_ids.format(v="e")}
            WITH e WHERE size(e.chunk_ids) = 0
            RETURN e.key AS k, COUNT {{ (e)-[:{self.R}]-() }} AS d""",
            ids=ids,
            rep=replace,
        )
        if dead:
            self._run(
                f"MATCH (e:{self.L}) WHERE e.key IN $keys DETACH DELETE e",
                keys=[r["k"] for r in dead],
            )
        return len(dead), edges + sum(r["d"] for r in dead)

    # ------------------------------------------------------------------ reads
    def _vertices(self, keys: list[str]) -> dict[str, Vertex]:
        rows = self._run(
            f"""MATCH (e:{self.L}) WHERE e.key IN $keys
            RETURN e.key AS id, e.name AS name, e.type AS type, e.description AS description,
                   e.aliases AS aliases, e.chunk_ids AS chunk_ids,
                   COUNT {{ (e)-[:{self.R}]-() }} AS degree""",
            keys=keys,
        )
        return {
            r["id"]: Vertex(
                id=r["id"],
                name=r["name"],
                type=r["type"],
                description=r["description"],
                aliases=sorted(r["aliases"]),
                chunk_ids=sorted(r["chunk_ids"]),
                degree=r["degree"],
            )
            for r in rows
        }

    def _subgraph(self, keys: list[str], hops: dict[str, int] | None = None) -> Subgraph:
        by_id = self._vertices(keys)
        rows = self._run(
            f"""MATCH (s:{self.L})-[r:{self.R}]->(t:{self.L})
            WHERE s.key IN $keys AND t.key IN $keys
            RETURN s.key AS s, r.relation AS r, t.key AS t, r.chunk_ids AS c
            ORDER BY s, r, t""",
            keys=keys,
        )
        edges = [Edge(x["s"], x["r"], x["t"], sorted(x["c"])) for x in rows]
        return Subgraph([by_id[k] for k in keys if k in by_id], edges, hops or {})

    def match_exact(self, text: str) -> list[str]:
        grams = sorted(query_ngrams(text))
        rows = self._run(
            f"""MATCH (e:{self.L})
            WHERE e.key IN $g OR any(a IN e.alias_keys WHERE a IN $g)
            RETURN DISTINCT e.key AS k ORDER BY k""",
            g=grams,
        )
        return [r["k"] for r in rows if len(r["k"]) > 1]

    def search_vectors(
        self, model: str, vector: np.ndarray, k: int, min_similarity: float
    ) -> list[tuple[str, float]]:
        meta = self._run(f"MATCH (m:{self.M}) RETURN m.vector_model AS model")
        if not meta or meta[0]["model"] != model or not self._vector_index_exists():
            return []
        rows = self._run(
            f"""CALL db.index.vector.queryNodes($index, $k, $v) YIELD node, score
            WHERE node:{self.L} RETURN node.key AS key, score ORDER BY score DESC, key""",
            index=self.index,
            k=k,
            v=np.asarray(vector, dtype=np.float32).tolist(),
        )
        out = [(r["key"], 2 * r["score"] - 1) for r in rows]  # (1 + cos) / 2 -> cos
        return [(key, sim) for key, sim in out if sim >= min_similarity]

    def neighbourhood(self, seeds: list[str], hops: int, limit: int = 300) -> Subgraph:
        present = set(self._vertices(seeds))
        dist: dict[str, int] = {}
        queue: deque[str] = deque()
        for s in seeds:
            if s in present and s not in dist:
                dist[s] = 0
                queue.append(s)
        while queue and len(dist) < limit:
            node = queue.popleft()
            if dist[node] >= hops:
                continue
            rows = self._run(
                f"""MATCH (:{self.L} {{key: $k}})-[:{self.R}]-(n:{self.L})
                RETURN DISTINCT n.key AS k ORDER BY k""",
                k=node,
            )
            for r in rows:
                if r["k"] not in dist and len(dist) < limit:
                    dist[r["k"]] = dist[node] + 1
                    queue.append(r["k"])
        return self._subgraph(list(dist), dist)

    def find(self, text: str, limit: int = 20) -> list[str]:
        needle = normalize_name(text)
        if not needle:
            return []
        rows = self._run(
            f"""MATCH (e:{self.L})
            WHERE e.key CONTAINS $t OR any(a IN e.alias_norms WHERE a CONTAINS $t)
            RETURN e.key AS k, e.key = $t AS exact, COUNT {{ (e)-[:{self.R}]-() }} AS d
            ORDER BY exact DESC, d DESC, k LIMIT $limit""",
            t=needle,
            limit=limit,
        )
        return [r["k"] for r in rows]

    def overview(self, types: list[str] | None, limit: int) -> Subgraph:
        rows = self._run(
            f"""MATCH (e:{self.L}) WHERE $types IS NULL OR e.type IN $types
            WITH e, COUNT {{ (e)-[:{self.R}]-() }} AS d
            RETURN e.key AS k ORDER BY d DESC, k LIMIT $limit""",
            types=types,
            limit=limit,
        )
        return self._subgraph([r["k"] for r in rows])

    def stats(self) -> dict[str, Any]:
        types = self._run(f"MATCH (e:{self.L}) RETURN e.type AS t, count(*) AS n")
        edges = self._run(f"MATCH (:{self.L})-[r:{self.R}]->(:{self.L}) RETURN count(r) AS n")
        counts = Counter({r["t"]: r["n"] for r in types})
        return {
            "vertices": sum(counts.values()),
            "edges": edges[0]["n"],
            "types": dict(sorted(counts.items())),
        }
