"use client";

// Stage-specific output views, keyed by stage id. Each receives the stage's ``data`` payload
// (shapes defined by the backend runners) and the whole run, for cross-stage views.
import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import { useState } from "react";
import { cn } from "cn";

import { EmbeddingScatter, TokenHistogram, ValueBar, type HistogramBin } from "@/components/pipeline/charts";
import { Q10View } from "@/components/pipeline/eval-views";
import { Q5View, Q6View, S8View } from "@/components/pipeline/graph-views";
import { Q2View, Q9View } from "@/components/pipeline/guard-views";
import { KeyValues } from "@/components/pipeline/KeyValues";
import { Table, td } from "@/components/pipeline/Table";
import { ChunkDialog } from "@/components/query/ChunkDialog";
import { Badge } from "@/components/ui/badge";
import type { RunState, StageId } from "@/lib/stages";

/* eslint-disable @typescript-eslint/no-explicit-any -- stage payloads are free-form JSON */
type Data = Record<string, any>;
export type StageView = (props: { data: Data; run: RunState }) => React.ReactNode;

const num = (n: number | undefined) => (n == null ? "" : n.toLocaleString());

function Preview({ text, className }: { text: string; className?: string }) {
  return <p className={cn("line-clamp-3 text-muted-foreground", className)}>{text}</p>;
}

const S1: StageView = ({ data }) => (
  <Table head={["File", "Size", "Pages", "Action", "Error"]}>
    {(data.files ?? []).map((f: Data) => (
      <tr key={f.document_id ?? f.filename}>
        <td className={td}>{f.filename}</td>
        <td className={td}>{num(Math.round(f.size_bytes / 1024))} KB</td>
        <td className={td}>{f.page_count}</td>
        <td className={td}><Badge variant={f.action === "duplicate" ? "secondary" : "outline"}>{f.action}</Badge></td>
        <td className={cn(td, "text-destructive")}>{f.error}</td>
      </tr>
    ))}
  </Table>
);

const S2: StageView = ({ data }) => (
  <div className="space-y-3">
    {(data.documents ?? []).map((d: Data) => (
      <div key={d.document_id} className="space-y-1 rounded border p-2 text-xs">
        <div className="flex flex-wrap gap-3">
          <span className="font-medium">{d.filename}</span>
          <span>{d.page_count} page(s)</span>
          <span>{num(d.raw_chars)} → {num(d.clean_chars)} chars</span>
          {d.error && <span className="text-destructive">{d.error}</span>}
        </div>
        {d.removed_header_footer_lines?.length > 0 && (
          <p>Removed repeated lines: {d.removed_header_footer_lines.map((l: string) => <Badge key={l} variant="secondary" className="mr-1">{l}</Badge>)}</p>
        )}
        <Preview text={d.clean_preview ?? ""} />
      </div>
    ))}
  </div>
);

const S3: StageView = ({ data }) => (
  <div className="space-y-2 text-xs">
    <Table head={["Document", "Tokens"]}>
      {(data.documents ?? []).map((d: Data) => (
        <tr key={d.document_id}><td className={td}>{d.filename}</td><td className={td}>{num(d.total_tokens)}</td></tr>
      ))}
    </Table>
    {data.preview?.tokens && (
      <div>
        <p className="mb-1 text-muted-foreground">Token boundaries (first {data.preview.tokens.length} tokens of the first document)</p>
        <div className="flex flex-wrap font-mono text-[11px] leading-5">
          {data.preview.tokens.map((t: Data, i: number) => (
            <span key={i} title={`id ${t.id}`} className={cn("whitespace-pre rounded-sm", i % 2 ? "bg-muted" : "bg-primary/10")}>
              {t.text.replace(/\n/g, "↵")}
            </span>
          ))}
        </div>
      </div>
    )}
  </div>
);

const S4: StageView = ({ data }) => (
  <div className="space-y-3 text-xs">
    {data.token_stats && <KeyValues values={{ chunks: data.chunk_count, ...data.token_stats }} />}
    {data.histogram && <TokenHistogram bins={data.histogram as HistogramBin[]} />}
    <Table head={["Chunk", "Pages", "Tokens", "Overlap chars", "Text"]}>
      {(data.chunks ?? []).map((c: Data) => (
        <tr key={c.chunk_id}>
          <td className={cn(td, "font-mono")}>{c.chunk_id}</td>
          <td className={td}>{c.page}{c.page_end !== c.page && `–${c.page_end}`}</td>
          <td className={td}>{c.token_count}</td>
          <td className={td}>{c.overlap_prev_chars}</td>
          <td className={td}><Preview text={c.text} /></td>
        </tr>
      ))}
    </Table>
    {data.chunks_shown < data.chunk_count && <p className="text-muted-foreground">Showing {data.chunks_shown} of {data.chunk_count} chunks.</p>}
  </div>
);

const S5: StageView = ({ data, run }) => {
  const s6 = run.stages.S6_segregate?.data as Data | undefined;
  const labels = Object.fromEntries((s6?.clusters ?? []).map((c: Data) => [c.id, c.label]));
  const points = (data.points ?? []).map((p: Data) => ({ ...p, cluster: s6?.assignments?.[p.chunk_id] }));
  return (
    <div className="space-y-3 text-xs">
      <KeyValues values={{ model: data.model, vectors: data.count, dimension: data.dimension, embed_ms: data.embed_ms, truncated: `${data.truncated_count} (limit ${data.max_input_tokens} model tokens)` }} />
      {points.length > 0 && <EmbeddingScatter points={points} labels={labels} />}
      {!s6 && <p className="text-muted-foreground">Cluster colours appear once S6 finishes.</p>}
    </div>
  );
};

const S6: StageView = ({ data }) => (
  <div className="space-y-3 text-xs">
    <KeyValues values={{ unique: data.unique_count, near_duplicates: data.duplicate_count, promoted_orphans: data.promoted_orphans }} />
    {data.label_fallback_reason && <p className="text-amber-700 dark:text-amber-500">Cluster labels from keywords: {data.label_fallback_reason}</p>}
    <Table head={["Cluster", "Label", "Source", "Size"]}>
      {(data.clusters ?? []).map((c: Data) => (
        <tr key={c.id}><td className={td}>{c.id}</td><td className={td}>{c.label}</td><td className={td}>{c.label_source}</td><td className={td}>{c.size}</td></tr>
      ))}
    </Table>
    {(data.duplicates ?? []).length > 0 && (
      <Table head={["Duplicate", "Of", "Similarity"]}>
        {data.duplicates.map((d: Data) => (
          <tr key={d.chunk_id}>
            <td className={td}><span className="font-mono">{d.chunk_id}</span><Preview text={d.text} /></td>
            <td className={td}><span className="font-mono">{d.duplicate_of}</span><Preview text={d.duplicate_of_text} /></td>
            <td className={td}>{d.similarity?.toFixed(3)}</td>
          </tr>
        ))}
      </Table>
    )}
  </div>
);

const S7: StageView = ({ data }) => (
  <div className="space-y-3 text-xs">
    <KeyValues values={{ collection: data.collection, inserted: data.inserted, total_vectors: data.total_vectors, deleted_previous_version: data.deleted_previous_version, metadata_updates: data.metadata_updates }} />
  </div>
);

const Q1: StageView = ({ data }) => (
  <pre className="overflow-auto rounded bg-muted p-2 font-mono text-[11px]">{JSON.stringify(data.request, null, 2)}</pre>
);

const Q3: StageView = ({ data }) => (
  <KeyValues values={{ token_count: data.token_count, embedding_ms: data.embedding_ms, dimension: data.dimension, model_loaded_now: data.model_loaded_now, vector_preview: (data.vector_preview ?? []).join(", ") + " …" }} />
);

function useCitationDialog() {
  const [open, setOpen] = useState<{ id: string; filename?: string } | null>(null);
  const dialog = <ChunkDialog chunkId={open?.id ?? null} filename={open?.filename} onClose={() => setOpen(null)} />;
  return { open: (id: string, filename?: string) => setOpen({ id, filename }), dialog };
}

const Q4: StageView = ({ data, run }) => {
  const threshold = run.stages.Q4_vector_retrieve?.params_used?.similarity_threshold as number | undefined;
  const { open, dialog } = useCitationDialog();
  return (
    <div className="space-y-2 text-xs">
      <KeyValues values={{ fetched: data.fetched, kept: data.kept, filtered: data.filtered, duplicates_collapsed: data.collapsed }} />
      <Table head={["#", "Chunk", "Similarity", "Text"]}>
        {(data.candidates ?? []).map((c: Data, i: number) => (
          <tr key={c.chunk_id} className={cn(c.filtered && "opacity-50")}>
            <td className={td}>{i + 1}</td>
            <td className={td}>
              <button className="font-mono whitespace-nowrap underline-offset-2 hover:underline" onClick={() => open(c.chunk_id, c.filename)}>{c.chunk_id}</button>
              <div className="text-muted-foreground">{c.filename} p. {c.page}</div>
              {c.collapsed_from?.length > 0 && <div className="text-muted-foreground">+{c.collapsed_from.length} duplicate(s) collapsed</div>}
            </td>
            <td className={cn(td, "w-40")}>
              <div className="mb-1 tabular-nums">{c.similarity.toFixed(3)}</div>
              <ValueBar value={c.similarity} threshold={threshold} muted={!!c.filtered} />
              {c.filtered && <div className="mt-1 text-muted-foreground">{c.filtered}</div>}
            </td>
            <td className={td}><Preview text={c.text} /></td>
          </tr>
        ))}
      </Table>
      {dialog}
    </div>
  );
};

function Movement({ before, after }: { before: number; after: number }) {
  const d = before - after;
  if (d === 0) return <Minus aria-label="unchanged" className="size-3.5 text-muted-foreground" />;
  const Icon = d > 0 ? ArrowUp : ArrowDown;
  return (
    <span className={cn("inline-flex items-center gap-0.5", d > 0 ? "text-emerald-600 dark:text-emerald-500" : "text-red-600 dark:text-red-500")}>
      <Icon aria-label={d > 0 ? "moved up" : "moved down"} className="size-3.5" />{Math.abs(d)}
    </span>
  );
}

const Q7: StageView = ({ data }) => (
  <Table head={["Before", "", "After", "Chunk", "Similarity", "Re-rank score", "Kept"]}>
    {(data.ranking ?? []).map((r: Data) => (
      <tr key={r.chunk_id} className={cn(!r.kept && "opacity-50")}>
        <td className={td}>{r.before_rank}</td>
        <td className={td}><Movement before={r.before_rank} after={r.after_rank} /></td>
        <td className={cn(td, "font-medium")}>{r.after_rank}</td>
        <td className={td}><span className="font-mono whitespace-nowrap">{r.chunk_id}</span><div className="text-muted-foreground">{r.filename} p. {r.page}</div></td>
        <td className={cn(td, "tabular-nums")}>{r.similarity.toFixed(3)}</td>
        <td className={cn(td, "w-36")}><div className="mb-1 tabular-nums">{r.score.toFixed(4)}</div><ValueBar value={r.score} /></td>
        <td className={td}>{r.kept ? "yes" : "no"}</td>
      </tr>
    ))}
  </Table>
);

const Q8: StageView = ({ data }) => (
  <div className="space-y-2 text-xs">
    <KeyValues values={{ llm_called: data.llm_called, model: data.model, prompt_tokens: data.prompt_tokens, completion_tokens: data.completion_tokens, citations: (data.citations ?? []).length }} />
    {data.unknown_citations?.length > 0 && (
      <p className="text-amber-700 dark:text-amber-500">Cited ids not in the context: {data.unknown_citations.join(", ")}</p>
    )}
    {data.prompt && (
      <details>
        <summary className="cursor-pointer text-muted-foreground">Final prompt sent to the LLM</summary>
        <div className="mt-2 space-y-2">
          {data.prompt.map((m: Data, i: number) => (
            <div key={i}>
              <Badge variant="outline">{m.role}</Badge>
              <pre className="mt-1 max-h-72 overflow-auto rounded bg-muted p-2 font-mono text-[11px] whitespace-pre-wrap">{m.content}</pre>
            </div>
          ))}
        </div>
      </details>
    )}
  </div>
);

export const STAGE_VIEWS: Partial<Record<StageId, StageView>> = {
  S1_upload: S1,
  S2_parse: S2,
  S3_tokenize: S3,
  S4_chunk: S4,
  S5_embed: S5,
  S6_segregate: S6,
  S7_vector_store: S7,
  S8_kg_build: S8View,
  Q1_validate: Q1,
  Q2_input_guardrail: Q2View,
  Q3_query_embed: Q3,
  Q4_vector_retrieve: Q4,
  Q5_graph_retrieve: Q5View,
  Q6_fusion: Q6View,
  Q7_rerank: Q7,
  Q8_generate: Q8,
  Q9_output_guardrail: Q9View,
  Q10_eval: Q10View,
};
