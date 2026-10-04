// Typed client for the FastAPI backend. Errors follow the backend's JEV envelope:
// { error: "VALIDATION_ERROR" | ..., details: [{ field, message }] }.
import type { ErrorResponse, RuntimeConfig } from "@/lib/schemas.generated";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export type FieldError = { field: string; message: string };

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    public details: FieldError[],
  ) {
    super(details.map((d) => (d.field ? `${d.field}: ${d.message}` : d.message)).join("; ") || code);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(0, "NETWORK_ERROR", [
      { field: "", message: `Cannot reach the backend at ${API_URL}` },
    ]);
  }
  if (!resp.ok) {
    const body = (await resp.json().catch(() => null)) as ErrorResponse | null;
    throw new ApiError(
      resp.status,
      body?.error ?? "HTTP_ERROR",
      body?.details ?? [{ field: "", message: resp.statusText }],
    );
  }
  return (await resp.json()) as T;
}

export type ComponentHealth = { status: string; detail: string; info: Record<string, string> };
export type Health = { status: "ok" | "degraded"; components: Record<string, ComponentHealth> };
export type PipelineDefaults = RuntimeConfig["defaults"];
export type RunSummary = {
  id: string;
  kind: "ingest" | "query";
  status: string;
  label: string;
  created_at: string;
  finished_at: string | null;
};
export type Chunk = {
  id: string;
  document_id: string;
  chunk_index: number;
  page: number;
  page_end: number;
  text: string;
  token_count: number;
  cluster_id: number;
  is_duplicate_of: string | null;
};

export type DocumentRow = {
  id: string;
  filename: string;
  extension: string;
  size_bytes: number;
  page_count: number;
  total_tokens: number;
  chunk_count: number;
  duplicate_chunk_count: number;
  chunk_size: number;
  chunk_overlap: number;
  version: number;
  status: string;
  error: string | null;
  created_at: string;
  updated_at: string;
};
export type ChunkPage = { items: Chunk[]; total: number; offset: number; limit: number };
export type ClusterRow = { id: number; label: string; label_source: string; size: number };
export type DocumentDeleted = {
  document_id: string;
  filename: string;
  chunks_deleted: number;
  vectors_deleted: number;
  duplicates_promoted: number;
  duplicates_repointed: number;
  vertices_removed: number;
  edges_removed: number;
  clusters_removed: number;
};

export type GraphData = {
  vertices: import("@/components/graph/GraphView").GraphVertex[];
  edges: import("@/components/graph/GraphView").GraphEdge[];
  seeds: string[];
  stats: { vertices: number; edges: number; types: Record<string, number> };
};

export type EvalDataset = {
  id: string;
  name: string;
  source: "upload" | "synthetic";
  status: "generating" | "ready" | "error";
  error: string | null;
  question_count: number;
  meta: Record<string, unknown>;
  created_at: string;
};
export type MetricMean = { mean: number | null; n: number };
export type EvalCellSummary = {
  cell: number;
  params: Record<string, number>;
  metrics: Record<string, MetricMean>;
  questions: number;
  statuses: Record<string, number>;
  mean_latency_ms: number;
  total_tokens: number;
  partial: boolean;
};
export type EvalRunRow = {
  id: string;
  dataset_id: string;
  dataset_name: string;
  status: "running" | "success" | "error" | "cancelled";
  param_grid: Record<string, number[]>;
  cells: Record<string, number>[];
  total: number;
  done: number;
  summary: EvalCellSummary[];
  error: string | null;
  phase: string | null;
  created_at: string;
  finished_at: string | null;
};
export type EvalEstimate = {
  cells: number;
  questions: number;
  query_calls_max: number;
  index_builds: { chunk_size: number; cached: boolean; chunks: number; max_calls: number }[];
  total_calls_max: number;
  missing_sources: string[];
};
export type EvalIndexRow = {
  id: string;
  chunk_size: number;
  chunk_overlap: number;
  dedup_threshold: number;
  build_graph: boolean;
  embedder: string;
  kg_model: string;
  status: "building" | "ready" | "error";
  error: string | null;
  chunk_count: number;
  vertex_count: number;
  edge_count: number;
  build_tokens: { calls?: number; prompt_tokens?: number; completion_tokens?: number };
  stale: boolean;
  created_at: string;
  finished_at: string | null;
};
export type EvalResultRow = {
  id: number;
  cell: number;
  question_id: number;
  question: string;
  ground_truth: string;
  status: string;
  answer: string;
  error: string | null;
  metrics: Record<string, number | null>;
  latency_ms: number;
  tokens: { calls?: number; cache_hits?: number; prompt_tokens?: number; completion_tokens?: number };
};
export type EvalResultDetail = EvalResultRow & {
  relevant_spans: { chunk_id?: string; filename?: string; page?: number }[];
  retrieved: { rank: number; chunk_id: string; filename: string | null; page: number | null; text: string; relevant: boolean | null }[];
  details: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any -- judge payloads
};
export type EvalLimits = {
  grid_params: string[];
  max_cells: number;
  max_grid_values: number;
  max_questions: number;
  max_synthetic: number;
  llm_calls_per_question: number;
  llm_calls_per_cell: number;
  metrics: string[];
};

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  config: () => request<RuntimeConfig>("/api/config"),
  saveConfig: (config: RuntimeConfig) =>
    request<RuntimeConfig>("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  runs: (kind: "ingest" | "query", limit = 30) =>
    request<{ items: RunSummary[]; total: number }>(`/api/runs?kind=${kind}&limit=${limit}`),
  graph: (q: { entity?: string; hops?: number; types?: string[]; limit?: number }) => {
    const p = new URLSearchParams();
    if (q.entity) p.set("entity", q.entity);
    if (q.hops != null) p.set("hops", String(q.hops));
    if (q.types?.length) p.set("types", q.types.join(","));
    if (q.limit) p.set("limit", String(q.limit));
    return request<GraphData>(`/api/graph?${p}`);
  },
  documents: () => request<DocumentRow[]>("/api/documents"),
  deleteDocument: (id: string) =>
    request<DocumentDeleted>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  chunks: (q: { document_id?: string; cluster_id?: number; duplicates?: boolean; offset?: number; limit?: number }) => {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(q)) if (v != null) p.set(k, String(v));
    return request<ChunkPage>(`/api/chunks?${p}`);
  },
  clusters: () => request<ClusterRow[]>("/api/clusters"),
  chunk: (id: string) => request<Chunk>(`/api/chunks/${encodeURIComponent(id)}`),
  ingest: (files: File[], params: Record<string, unknown>) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("params", JSON.stringify(params));
    return request<{ job_id: string }>("/api/ingest", { method: "POST", body: form });
  },
  evalLimits: () => request<EvalLimits>("/api/eval/limits"),
  evalDatasets: () => request<EvalDataset[]>("/api/eval/datasets"),
  uploadEvalDataset: (name: string, file: File) => {
    const form = new FormData();
    form.append("name", name);
    form.append("file", file);
    return request<EvalDataset>("/api/eval/datasets", { method: "POST", body: form });
  },
  syntheticEvalDataset: (body: { name: string; count: number; seed: number }) =>
    request<EvalDataset>("/api/eval/datasets/synthetic", json(body)),
  evalRuns: () => request<EvalRunRow[]>("/api/eval/runs"),
  evalEstimate: (body: { dataset_id: string; param_grid: Record<string, number[]> }) =>
    request<EvalEstimate>("/api/eval/estimate", json(body)),
  evalIndexes: () => request<EvalIndexRow[]>("/api/eval/indexes"),
  deleteEvalIndex: (id: string) =>
    request<{ deleted: boolean }>(`/api/eval/indexes/${encodeURIComponent(id)}`, { method: "DELETE" }),
  startEvalRun: (body: { dataset_id: string; param_grid: Record<string, number[]> }) =>
    request<{ eval_run_id: string }>("/api/eval/runs", json(body)),
  evalRun: (id: string, cell: number) =>
    request<EvalRunRow & { results: EvalResultRow[] }>(`/api/eval/runs/${encodeURIComponent(id)}?cell=${cell}`),
  evalResult: (id: number) => request<EvalResultDetail>(`/api/eval/results/${id}`),
  cancelEvalRun: (id: string) =>
    request<{ cancelled: boolean }>(`/api/eval/runs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  query: (body: { query: string; params?: Record<string, number | boolean | undefined> }) =>
    request<{ run_id: string }>("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export function eventsUrl(kind: "ingest" | "query", runId: string): string {
  return `${API_URL}/api/${kind}/${encodeURIComponent(runId)}/events`;
}
