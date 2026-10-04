"use client";

// Output views for the knowledge-graph stages: S8 build, Q5 graph retrieval, Q6 fusion.
import { cn } from "cn";

import { GraphView } from "@/components/graph/GraphView";
import { KeyValues } from "@/components/pipeline/KeyValues";
import { Table, td } from "@/components/pipeline/Table";
import { Badge } from "@/components/ui/badge";
import type { RunState } from "@/lib/stages";

/* eslint-disable @typescript-eslint/no-explicit-any -- stage payloads are free-form JSON */
type Data = Record<string, any>;
type Props = { data: Data; run: RunState };

export function S8View({ data }: Props) {
  const values: Record<string, unknown> = {
    graph: `${data.vertex_count} vertices, ${data.edge_count} edges`,
    removed_from_previous_version: `${data.removed_vertices ?? 0} vertices, ${data.removed_edges ?? 0} edges`,
  };
  if (data.chunks_processed != null) {
    values.extracted = `${data.entities_extracted} entities, ${data.relations_extracted} relations from ${data.chunks_processed} chunk(s)`;
    values.added = `+${data.added_vertices} vertices, +${data.added_edges} edges (${data.merged_vertices} merged mentions)`;
    values.name_vectors_added = data.vectors_added;
  }
  return (
    <div className="space-y-3 text-xs">
      <KeyValues values={values} />
      {data.types && Object.keys(data.types).length > 0 && (
        <div className="flex flex-wrap gap-1">
          {Object.entries(data.types as Record<string, number>).map(([t, n]) => (
            <Badge key={t} variant="outline">
              {t.toLowerCase()} {n}
            </Badge>
          ))}
        </div>
      )}
      {data.failures?.length > 0 && (
        <Table head={["Chunk", "Extraction error"]}>
          {data.failures.map((f: Data) => (
            <tr key={f.chunk_id}>
              <td className={cn(td, "font-mono whitespace-nowrap")}>{f.chunk_id}</td>
              <td className={cn(td, "text-destructive")}>{f.error}</td>
            </tr>
          ))}
        </Table>
      )}
      {data.sample?.vertices?.length > 0 && (
        <div>
          <p className="mb-1 text-muted-foreground">Entities extracted in this run (up to 150)</p>
          <GraphView vertices={data.sample.vertices} edges={data.sample.edges} height={300} />
        </div>
      )}
    </div>
  );
}

export function Q5View({ data }: Props) {
  return (
    <div className="space-y-3 text-xs">
      <p className="text-muted-foreground">
        {data.extraction_note}
        {data.query_entities?.length > 0 && ": "}
        {(data.query_entities ?? []).map((e: string) => (
          <Badge key={e} variant="secondary" className="mr-1">
            {e}
          </Badge>
        ))}
      </p>
      {data.matches?.length > 0 ? (
        <Table head={["Matched entity", "Method", "Score", "Probe"]}>
          {data.matches.map((m: Data) => (
            <tr key={m.id}>
              <td className={td}>{m.id}</td>
              <td className={td}>
                <Badge variant={m.method === "exact" ? "default" : "outline"}>{m.method}</Badge>
              </td>
              <td className={cn(td, "tabular-nums")}>{m.score.toFixed(3)}</td>
              <td className={cn(td, "text-muted-foreground")}>
                {m.method === "exact" ? "name/alias found in the query" : m.probe}
              </td>
            </tr>
          ))}
        </Table>
      ) : (
        <p className="text-muted-foreground">
          No vertex matched exactly or above the {data.match_threshold} embedding threshold.
        </p>
      )}
      {data.subgraph?.vertices?.length > 0 && (
        <div>
          <p className="mb-1 text-muted-foreground">Traversed sub-graph (ringed: matched entities)</p>
          <GraphView vertices={data.subgraph.vertices} edges={data.subgraph.edges} height={300} />
        </div>
      )}
      {data.triples?.length > 0 && (
        <details>
          <summary className="cursor-pointer text-muted-foreground">
            Relation triples sent to the LLM ({data.triples.length})
          </summary>
          <ul className="mt-1 space-y-0.5">
            {data.triples.map((t: Data, i: number) => (
              <li key={i}>
                {t.source} <span className="text-muted-foreground">— {t.relation} →</span> {t.target}{" "}
                <span className="text-muted-foreground">(hop {t.hop})</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {data.evidence?.length > 0 && (
        <Table head={["Evidence chunk", "Graph score", "Via entities"]}>
          {data.evidence.slice(0, 20).map((e: Data) => (
            <tr key={e.chunk_id}>
              <td className={cn(td, "font-mono whitespace-nowrap")}>{e.chunk_id}</td>
              <td className={cn(td, "tabular-nums")}>{e.score}</td>
              <td className={cn(td, "text-muted-foreground")}>{e.via.join(", ")}</td>
            </tr>
          ))}
        </Table>
      )}
    </div>
  );
}

export function Q6View({ data, run }: Props) {
  const w = run.stages.Q6_fusion?.params_used?.hybrid_weight_vector as number | undefined;
  return (
    <div className="space-y-2 text-xs">
      {w != null && (
        <p className="text-muted-foreground">
          fused score = {w} / (60 + vector rank) + {(1 - w).toFixed(2)} / (60 + graph rank); the top K
          rows go on to re-ranking (greyed rows do not)
        </p>
      )}
      <Table head={["Fused", "Chunk", "Source", "Vector rank", "Graph rank", "Fused score"]}>
        {(data.rows ?? []).map((r: Data) => (
          <tr
            key={r.chunk_id}
            className={cn(r.kept === false && "opacity-50")}
            title={r.kept === false ? "below Top K: not passed to re-ranking" : undefined}
          >
            <td className={cn(td, "font-medium")}>{r.fused_rank}</td>
            <td className={td}>
              <span className="font-mono whitespace-nowrap">{r.chunk_id}</span>
              <div className="text-muted-foreground">
                {r.filename} p. {r.page}
              </div>
            </td>
            <td className={td}>
              <Badge variant={r.source === "both" ? "default" : "outline"}>{r.source}</Badge>
            </td>
            <td className={cn(td, "tabular-nums")}>{r.vector_rank ?? "–"}</td>
            <td className={cn(td, "tabular-nums")}>{r.graph_rank ?? "–"}</td>
            <td className={cn(td, "tabular-nums")}>{r.fused_score.toFixed(5)}</td>
          </tr>
        ))}
      </Table>
    </div>
  );
}
