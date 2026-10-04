"use client";

// Eval page: golden sets (upload / synthetic), a parameter-grid run builder with an LLM-call
// estimate, the run list, and a run's metrics table, charts and per-question drill-down.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Sparkles, Square, Trash2, Upload } from "lucide-react";
import { useDeferredValue, useState } from "react";
import { cn } from "cn";

import { MetricsChart, toggleCompared, type Compared } from "@/components/eval/MetricsChart";
import { MetricsTable } from "@/components/eval/MetricsTable";
import { Header } from "@/components/layout/Header";
import { ValueBar } from "@/components/pipeline/charts";
import { td } from "@/components/pipeline/Table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { ApiError, api, type EvalDataset, type EvalLimits, type EvalRunRow, type FieldError } from "@/lib/api";
import { cellLabel, fmtMetric, METRICS, parseGrid, varyingKeys } from "@/lib/eval";

/* eslint-disable @typescript-eslint/no-explicit-any -- judge payloads are free-form JSON */

const SELECT = "h-8 rounded-md border bg-background px-2 text-xs";
const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  ready: "secondary",
  success: "secondary",
  generating: "outline",
  running: "outline",
  error: "destructive",
  cancelled: "outline",
};

function Panel({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={cn("space-y-3 rounded-lg border bg-card p-4", className)}>
      <h2 className="text-sm font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function Errors({ errors }: { errors: FieldError[] }) {
  return errors.map((e) => (
    <p key={e.field + e.message} className="text-xs text-destructive">
      {e.field ? `${e.field}: ${e.message}` : e.message}
    </p>
  ));
}

const toErrors = (e: unknown): FieldError[] => (e instanceof ApiError ? e.details : [{ field: "", message: String(e) }]);

// ---------------------------------------------------------------- golden sets
function Datasets({ datasets, limits }: { datasets: EvalDataset[]; limits?: EvalLimits }) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [count, setCount] = useState(10);
  const [seed, setSeed] = useState(42);
  const [errors, setErrors] = useState<FieldError[]>([]);
  const done = () => {
    setErrors([]);
    setName("");
    setFile(null);
    queryClient.invalidateQueries({ queryKey: ["eval", "datasets"] });
  };
  const upload = useMutation({ mutationFn: () => api.uploadEvalDataset(name.trim(), file!), onSuccess: done, onError: (e) => setErrors(toErrors(e)) });
  const generate = useMutation({
    mutationFn: () => api.syntheticEvalDataset({ name: name.trim(), count, seed }),
    onSuccess: done,
    onError: (e) => setErrors(toErrors(e)),
  });
  return (
    <Panel title="Golden sets">
      {datasets.length ? (
        <ul className="divide-y rounded border text-xs">
          {datasets.map((d) => (
            <li key={d.id} className="flex flex-wrap items-center gap-2 px-2 py-1.5">
              <span className="font-medium">{d.name}</span>
              <Badge variant="outline">{d.source}</Badge>
              <Badge variant={STATUS_VARIANT[d.status]}>{d.status}</Badge>
              <span className="text-muted-foreground">{d.question_count} question(s)</span>
              {d.error && <span className="text-destructive">{d.error}</span>}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-xs text-muted-foreground">No golden sets yet.</p>
      )}
      <div className="space-y-2">
        <div className="space-y-1">
          <Label htmlFor="ds-name" className="text-xs">Name</Label>
          <Input id="ds-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={100} className="h-8 text-xs" />
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <div className="space-y-1">
            <Label htmlFor="ds-file" className="text-xs">JSONL file</Label>
            <Input id="ds-file" type="file" accept=".jsonl,.json,.txt" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="h-8 text-xs" />
          </div>
          <Button size="sm" disabled={!name.trim() || !file || upload.isPending} onClick={() => upload.mutate()}>
            <Upload /> Upload
          </Button>
        </div>
        <p className="text-[11px] text-muted-foreground">
          {'One JSON object per line: {"question", "ground_truth", "relevant_chunk_ids"?}. Chunk ids are stored as text spans, so they survive re-ingestion.'}
        </p>
        <div className="flex flex-wrap items-end gap-2">
          <div className="space-y-1">
            <Label htmlFor="ds-count" className="text-xs">Questions</Label>
            <Input id="ds-count" type="number" min={1} max={limits?.max_synthetic ?? 50} value={count} onChange={(e) => setCount(Number(e.target.value))} className="h-8 w-20 text-xs" />
          </div>
          <div className="space-y-1">
            <Label htmlFor="ds-seed" className="text-xs">Seed</Label>
            <Input id="ds-seed" type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="h-8 w-24 text-xs" />
          </div>
          <Button size="sm" variant="outline" disabled={!name.trim() || generate.isPending} onClick={() => generate.mutate()}>
            <Sparkles /> Generate from chunks
          </Button>
        </div>
        <p className="text-[11px] text-muted-foreground">Synthetic: one LLM call per question, each from a seeded random unique chunk.</p>
        <Errors errors={errors} />
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------- new run
function NewRun({ datasets, limits, onStarted }: { datasets: EvalDataset[]; limits?: EvalLimits; onStarted: (id: string) => void }) {
  const config = useQuery({ queryKey: ["config"], queryFn: api.config });
  const ready = datasets.filter((d) => d.status === "ready");
  const [datasetId, setDatasetId] = useState("");
  const [input, setInput] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<FieldError[]>([]);
  // Pin the default once, so a set that becomes ready later (e.g. a synthetic one) can't take over.
  if (!datasetId && ready[0]) setDatasetId(ready[0].id);
  const dataset = ready.find((d) => d.id === datasetId) ?? ready[0];
  const { grid, errors: gridErrors, cells } = parseGrid(input, limits?.max_grid_values ?? 6);
  const tooMany = limits ? cells > limits.max_cells : false;
  const gridOk = !!dataset && !tooMany && Object.keys(gridErrors).length === 0;
  // The server prices the run: per-question calls plus any throwaway index builds.
  const request = useDeferredValue(JSON.stringify({ dataset_id: dataset?.id, param_grid: grid }));
  const estimate = useQuery({
    queryKey: ["eval", "estimate", request],
    queryFn: () => api.evalEstimate(JSON.parse(request)),
    enabled: gridOk,
    retry: false,
  });
  const est = gridOk ? estimate.data : undefined;
  const missing = est?.missing_sources ?? [];
  const start = useMutation({
    mutationFn: () => api.startEvalRun({ dataset_id: dataset!.id, param_grid: grid }),
    onSuccess: (r) => {
      setErrors([]);
      onStarted(r.eval_run_id);
    },
    onError: (e) => setErrors(toErrors(e)),
  });
  const defaults = (config.data?.defaults ?? {}) as Record<string, number>;
  return (
    <Panel title="New eval run">
      {ready.length === 0 ? (
        <p className="text-xs text-muted-foreground">Add a golden set first.</p>
      ) : (
        <>
          <div className="space-y-1">
            <Label htmlFor="run-dataset" className="text-xs">Golden set</Label>
            <select id="run-dataset" className={cn(SELECT, "w-full")} value={dataset?.id} onChange={(e) => setDatasetId(e.target.value)}>
              {ready.map((d) => (
                <option key={d.id} value={d.id}>{d.name} ({d.question_count})</option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {(limits?.grid_params ?? []).map((p) => (
              <div key={p} className="space-y-1">
                <Label htmlFor={`grid-${p}`} className="text-xs">{p}</Label>
                <Input
                  id={`grid-${p}`}
                  value={input[p] ?? ""}
                  placeholder={defaults[p] != null ? `default ${defaults[p]}` : ""}
                  onChange={(e) => setInput({ ...input, [p]: e.target.value })}
                  aria-invalid={!!gridErrors[p]}
                  className="h-8 text-xs"
                />
                {gridErrors[p] && <p className="text-[11px] text-destructive">{gridErrors[p]}</p>}
              </div>
            ))}
          </div>
          <p className="text-[11px] text-muted-foreground">
            Comma-separated values per parameter; every combination is one grid cell. With chunk_size in the grid, every cell
            queries a throwaway index re-built from your original uploads (cached for later runs); the live index is untouched.
          </p>
          <div className={cn("space-y-0.5 text-xs", tooMany && "text-destructive")}>
            <p>
              {cells} cell(s) × {dataset?.question_count ?? 0} question(s)
              {tooMany && ` — at most ${limits?.max_cells} cells.`}
            </p>
            {est && (
              <>
                <p>
                  At most <strong>{est.total_calls_max.toLocaleString()}</strong> LLM calls: {est.query_calls_max.toLocaleString()} for
                  the questions (worst case: every answer regenerated and re-judged; cached calls cost nothing)
                  {est.index_builds.length > 0 && " plus index builds:"}
                </p>
                {est.index_builds.length > 0 && (
                  <ul className="list-disc pl-5 text-muted-foreground">
                    {est.index_builds.map((b) => (
                      <li key={b.chunk_size}>
                        chunk_size {b.chunk_size}:{" "}
                        {b.cached
                          ? "cached index, no calls"
                          : `new index, ${b.chunks.toLocaleString()} chunks → up to ${b.max_calls.toLocaleString()} calls (graph extraction + cluster labels)`}
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
            {gridOk && estimate.isError && <Errors errors={toErrors(estimate.error)} />}
            {missing.length > 0 && (
              <p className="text-destructive">
                Original uploads were not kept for {missing.join(", ")} (ingested before they were saved). Upload these files
                again on the Ingest page with the same settings: they are only stored, not re-ingested.
              </p>
            )}
          </div>
          <Button size="sm" disabled={!gridOk || !est || missing.length > 0 || start.isPending} onClick={() => start.mutate()}>
            <Play /> Run evaluation
          </Button>
          <Errors errors={errors} />
        </>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- run detail
function ResultSheet({ resultId, onClose }: { resultId: number | null; onClose: () => void }) {
  const result = useQuery({
    queryKey: ["eval", "result", resultId],
    queryFn: () => api.evalResult(resultId!),
    enabled: resultId != null,
  });
  const r = result.data;
  const d = (r?.details ?? {}) as Record<string, any>;
  return (
    <Sheet open={resultId != null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-2xl">
        <SheetHeader>
          <SheetTitle>Question drill-down</SheetTitle>
          <SheetDescription>{r?.question ?? "Loading…"}</SheetDescription>
        </SheetHeader>
        {r && (
          <div className="flex-1 space-y-4 overflow-y-auto px-4 pb-4 text-xs">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-4">
              {METRICS.map((m) => (
                <div key={m.id} title={m.help}>
                  <div className="text-muted-foreground">{m.label}</div>
                  <div className="font-semibold tabular-nums">{fmtMetric(r.metrics[m.id])}</div>
                </div>
              ))}
            </div>
            <div className="space-y-1">
              <h4 className="font-medium">Reference answer</h4>
              <p className="whitespace-pre-wrap">{r.ground_truth}</p>
            </div>
            <div className="space-y-1">
              <h4 className="font-medium">Answer <Badge variant={STATUS_VARIANT[r.status] ?? "outline"}>{r.status}</Badge></h4>
              <p className="whitespace-pre-wrap">{r.answer || r.error}</p>
            </div>
            <div className="space-y-1">
              <h4 className="font-medium">Retrieved context ({r.retrieved.length})</h4>
              {r.relevant_spans.length > 0 && (
                <p className="text-muted-foreground">
                  Relevant: {r.relevant_spans.map((s) => `${s.filename} p. ${s.page}`).join(", ")}
                </p>
              )}
              <ol className="space-y-1">
                {r.retrieved.map((c) => {
                  const verdict = (d.context_precision?.verdicts ?? []).find((v: any) => v.chunk_id === c.chunk_id);
                  return (
                    <li key={c.chunk_id} className="space-y-0.5 rounded border p-2">
                      <div className="flex flex-wrap gap-2">
                        <span className="font-medium">#{c.rank}</span>
                        <span>{c.filename} p. {c.page}</span>
                        <span className="font-mono text-muted-foreground">{c.chunk_id}</span>
                        {c.relevant != null && <Badge variant={c.relevant ? "secondary" : "outline"}>{c.relevant ? "golden match" : "not golden"}</Badge>}
                        {verdict && <Badge variant="outline">judge: {verdict.relevant ? "relevant" : "not relevant"}</Badge>}
                      </div>
                      {verdict?.reason && <p className="text-muted-foreground">Judge: {verdict.reason}</p>}
                      <details>
                        <summary className="cursor-pointer text-muted-foreground">Text</summary>
                        <p className="mt-1 whitespace-pre-wrap">{c.text}</p>
                      </details>
                    </li>
                  );
                })}
              </ol>
            </div>
            <JudgeList title="Context recall: reference statements" rows={d.context_recall?.statements} ok="attributed" note={d.context_recall?.note} />
            {d.answer_correctness && (
              <div className="space-y-1">
                <h4 className="font-medium">
                  Answer correctness: F1 {fmtMetric(d.answer_correctness.f1)}, similarity {fmtMetric(d.answer_correctness.similarity)}
                </h4>
                {d.answer_correctness.note && <p className="text-muted-foreground">{d.answer_correctness.note}</p>}
                {(["tp", "fp", "fn"] as const).map((k) =>
                  d.answer_correctness[k]?.length ? (
                    <div key={k}>
                      <span className="font-medium">{{ tp: "Correct (TP)", fp: "Contradicts reference (FP)", fn: "Missed (FN)" }[k]}: </span>
                      <span className="text-muted-foreground">{d.answer_correctness[k].join(" · ")}</span>
                    </div>
                  ) : null,
                )}
              </div>
            )}
            <JudgeList title="Faithfulness: answer sentences" rows={d.faithfulness?.sentences} ok="supported" note={d.faithfulness?.note} />
            <p className="text-muted-foreground">
              {r.tokens.calls ?? 0} LLM call(s), {((r.tokens.prompt_tokens ?? 0) + (r.tokens.completion_tokens ?? 0)).toLocaleString()} tokens
              {r.tokens.cache_hits ? `, ${r.tokens.cache_hits} cached` : ""} · {r.latency_ms.toLocaleString()} ms
            </p>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}

function JudgeList({ title, rows, ok, note }: { title: string; rows?: any[]; ok: string; note?: string }) {
  if (!rows?.length && !note) return null;
  return (
    <div className="space-y-1">
      <h4 className="font-medium">{title}</h4>
      {note && <p className="text-muted-foreground">{note}</p>}
      <ul className="space-y-0.5">
        {(rows ?? []).map((s, i) => (
          <li key={i} className="flex gap-2">
            <span className="w-24 shrink-0">{s[ok] ? ok : `not ${ok}`}</span>
            <span>{s.text}</span>
            {s.reason && <span className="text-muted-foreground">({s.reason})</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

function RunDetail({ runId }: { runId: string }) {
  const queryClient = useQueryClient();
  const [cell, setCell] = useState(0);
  const [compared, setCompared] = useState<Compared>([{ cell: 0, slot: 1 }]);
  const [resultId, setResultId] = useState<number | null>(null);
  const run = useQuery({
    queryKey: ["eval", "run", runId, cell],
    queryFn: () => api.evalRun(runId, cell),
    refetchInterval: (q) => (q.state.data?.status === "running" ? 2000 : false),
  });
  const cancel = useMutation({
    mutationFn: () => api.cancelEvalRun(runId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["eval"] }),
  });
  const r = run.data;
  if (!r) return <p className="text-xs text-muted-foreground">{run.isError ? run.error.message : "Loading run…"}</p>;
  const varying = varyingKeys(r.cells);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge variant={STATUS_VARIANT[r.status]}>{r.status}</Badge>
        <span>{r.dataset_name}</span>
        <span className="text-muted-foreground">
          {r.done} / {r.total} question runs · {r.cells.length} cell(s)
        </span>
        {r.status === "running" && (
          <>
            <div className="w-40"><ValueBar value={r.total ? r.done / r.total : 0} /></div>
            <Button size="xs" variant="outline" onClick={() => cancel.mutate()} disabled={cancel.isPending}>
              <Square /> Cancel
            </Button>
          </>
        )}
        {r.phase && <span className="text-muted-foreground">{r.phase}…</span>}
        {r.error && <span className="text-destructive">{r.error}</span>}
      </div>
      {r.summary.length > 0 ? (
        <>
          <MetricsTable summary={r.summary} varying={varying} selected={cell} onSelect={setCell} />
          {r.summary.length > 1 && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="text-muted-foreground">Compare (up to 4):</span>
              {r.summary.map((s) => {
                const on = compared.find((c) => c.cell === s.cell);
                return (
                  <label key={s.cell} className="flex items-center gap-1">
                    <input type="checkbox" checked={!!on} onChange={() => setCompared(toggleCompared(compared, s.cell))} />
                    {on && <span aria-hidden className="inline-block size-2.5 rounded-sm" style={{ background: `var(--series-${on.slot})` }} />}
                    {cellLabel(s.params, varying)}
                  </label>
                );
              })}
            </div>
          )}
          <MetricsChart summary={r.summary} compared={compared} varying={varying} />
        </>
      ) : (
        <p className="text-xs text-muted-foreground">Metrics appear when the first grid cell finishes.</p>
      )}
      <div className="space-y-1">
        <h3 className="text-xs font-semibold">
          Questions · {r.cells[cell] ? cellLabel(r.cells[cell], varying) : ""}
          {r.cells.length > 1 && <span className="font-normal text-muted-foreground"> (click a table row to switch cell)</span>}
        </h3>
        <div className="max-h-96 overflow-auto rounded border">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-muted text-left text-muted-foreground">
              <tr>
                <th className="px-2 py-1 font-medium">Question</th>
                <th className="px-2 py-1 font-medium">Status</th>
                {METRICS.map((m) => <th key={m.id} className="px-2 py-1 font-medium" title={m.help}>{m.label}</th>)}
              </tr>
            </thead>
            <tbody className="divide-y">
              {r.results.map((q) => (
                <tr key={q.id} className="cursor-pointer hover:bg-muted/50" onClick={() => setResultId(q.id)}>
                  <td className={cn(td, "max-w-80")}>{q.question}</td>
                  <td className={td}><Badge variant={STATUS_VARIANT[q.status] ?? "outline"}>{q.status}</Badge></td>
                  {METRICS.map((m) => <td key={m.id} className={cn(td, "tabular-nums")}>{fmtMetric(q.metrics[m.id])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <ResultSheet resultId={resultId} onClose={() => setResultId(null)} />
    </div>
  );
}

// ---------------------------------------------------------------- throwaway indexes
function Indexes({ runActive }: { runActive: boolean }) {
  const queryClient = useQueryClient();
  const list = useQuery({
    queryKey: ["eval", "indexes"],
    queryFn: api.evalIndexes,
    refetchInterval: runActive ? 3000 : false,
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteEvalIndex(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["eval"] }),
  });
  const rows = list.data ?? [];
  return (
    <Panel title="Throwaway indexes (chunk_size experiments)">
      {rows.length === 0 ? (
        <p className="text-xs text-muted-foreground">None yet. Runs with chunk_size in the grid build one per chunk size.</p>
      ) : (
        <div className="overflow-x-auto rounded border">
          <table className="w-full text-xs">
            <thead className="bg-muted text-left text-muted-foreground">
              <tr>
                {["chunk_size", "Status", "Chunks", "Graph", "Build LLM calls", "Built", ""].map((h) => (
                  <th key={h} className="px-2 py-1 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y">
              {rows.map((i) => (
                <tr key={i.id}>
                  <td className={cn(td, "tabular-nums")}>
                    {i.chunk_size} <span className="text-muted-foreground">(overlap {i.chunk_overlap})</span>
                  </td>
                  <td className={td}>
                    <Badge variant={STATUS_VARIANT[i.status] ?? "outline"}>{i.status}</Badge>
                    {i.stale && (
                      <Badge variant="outline" className="ml-1" title="Built from an older corpus; new runs build a fresh index">
                        stale
                      </Badge>
                    )}
                    {i.error && <span className="ml-1 text-destructive">{i.error}</span>}
                  </td>
                  <td className={cn(td, "tabular-nums")}>{i.chunk_count.toLocaleString()}</td>
                  <td className={cn(td, "tabular-nums")}>
                    {i.vertex_count} vertices, {i.edge_count} edges
                  </td>
                  <td className={cn(td, "tabular-nums")}>
                    {i.build_tokens.calls ?? 0} (
                    {((i.build_tokens.prompt_tokens ?? 0) + (i.build_tokens.completion_tokens ?? 0)).toLocaleString()} tokens)
                  </td>
                  <td className={td}>{new Date(i.created_at).toLocaleString()}</td>
                  <td className={td}>
                    <Button
                      size="xs"
                      variant="ghost"
                      aria-label={`Delete chunk_size ${i.chunk_size} index`}
                      disabled={runActive || remove.isPending}
                      onClick={() => remove.mutate(i.id)}
                    >
                      <Trash2 />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {remove.isError && <Errors errors={toErrors(remove.error)} />}
    </Panel>
  );
}

// ---------------------------------------------------------------- page
function RunList({ runs, selected, onSelect }: { runs: EvalRunRow[]; selected: string | null; onSelect: (id: string) => void }) {
  if (!runs.length) return <p className="text-xs text-muted-foreground">No eval runs yet.</p>;
  return (
    <ul className="divide-y rounded border text-xs">
      {runs.map((r) => (
        <li key={r.id}>
          <button
            type="button"
            onClick={() => onSelect(r.id)}
            className={cn("flex w-full flex-wrap items-center gap-2 px-2 py-1.5 text-left hover:bg-muted/50", selected === r.id && "bg-muted")}
          >
            <Badge variant={STATUS_VARIANT[r.status]}>{r.status}</Badge>
            <span className="font-medium">{r.dataset_name}</span>
            <span className="text-muted-foreground">
              {r.cells.length} cell(s) · {r.done}/{r.total} · {new Date(r.created_at).toLocaleString()}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

export function EvalView() {
  const queryClient = useQueryClient();
  const limits = useQuery({ queryKey: ["eval", "limits"], queryFn: api.evalLimits, staleTime: Infinity });
  const datasets = useQuery({
    queryKey: ["eval", "datasets"],
    queryFn: api.evalDatasets,
    refetchInterval: (q) => (q.state.data?.some((d) => d.status === "generating") ? 2000 : false),
  });
  const runs = useQuery({
    queryKey: ["eval", "runs"],
    queryFn: api.evalRuns,
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === "running") ? 3000 : false),
  });
  const [selected, setSelected] = useState<string | null>(null);
  const current = selected ?? runs.data?.[0]?.id ?? null;
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <Header title="Eval" />
      <main className="flex-1 space-y-4 overflow-y-auto p-4">
        {(datasets.isError || runs.isError) && (
          <p className="text-xs text-destructive">{(datasets.error ?? runs.error)?.message}</p>
        )}
        <div className="grid gap-4 lg:grid-cols-2">
          <Datasets datasets={datasets.data ?? []} limits={limits.data} />
          <NewRun
            datasets={datasets.data ?? []}
            limits={limits.data}
            onStarted={(id) => {
              setSelected(id);
              queryClient.invalidateQueries({ queryKey: ["eval", "runs"] });
            }}
          />
        </div>
        <Panel title="Runs">
          <RunList runs={runs.data ?? []} selected={current} onSelect={setSelected} />
          {current && <RunDetail key={current} runId={current} />}
        </Panel>
        <Indexes runActive={!!runs.data?.some((r) => r.status === "running")} />
      </main>
    </div>
  );
}
