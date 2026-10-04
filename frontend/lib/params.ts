// Parameter metadata for the Parameters Panel plus validation. Ranges mirror the backend; the
// generated zod schemas are the source of truth for validation (lib/params.test.ts checks the
// slider ranges against them). JSON Schema cannot express cross-field rules, so those are here.
import { IngestParamsSchema, QueryParamsSchema } from "@/lib/schemas.generated";

export type ParamKey =
  | "chunk_size"
  | "chunk_overlap"
  | "dedup_threshold"
  | "top_k"
  | "top_n"
  | "similarity_threshold"
  | "temperature"
  | "seed"
  | "graph_hops"
  | "hybrid_weight_vector"
  | "build_graph";

export type ParamMeta = {
  key: ParamKey;
  label: string;
  min: number;
  max: number;
  step: number;
  description: string;
  kind?: "number" | "switch"; // number input / on-off switch instead of a slider
};

export const INGEST_PARAMS: ParamMeta[] = [
  { key: "chunk_size", label: "Chunk size", min: 128, max: 2048, step: 16, description: "Tokens per chunk" },
  { key: "chunk_overlap", label: "Chunk overlap", min: 0, max: 1024, step: 8, description: "Tokens shared by neighbouring chunks; at most chunk size / 2" },
  { key: "dedup_threshold", label: "Dedup threshold", min: 0.8, max: 1, step: 0.01, description: "Similarity above which a chunk is flagged as a near-duplicate" },
  { key: "build_graph", label: "Build knowledge graph", min: 0, max: 1, step: 1, kind: "switch", description: "Run S8: LLM entity/relation extraction for new chunks" },
];

export const QUERY_PARAMS: ParamMeta[] = [
  { key: "top_k", label: "Top K", min: 1, max: 50, step: 1, description: "Candidates fetched from Chroma" },
  { key: "top_n", label: "Top N", min: 1, max: 20, step: 1, description: "Chunks kept after re-ranking and sent to the LLM; at most Top K" },
  { key: "similarity_threshold", label: "Similarity threshold", min: 0, max: 1, step: 0.01, description: "Cosine cut-off; weaker chunks are dropped and greyed out" },
  { key: "temperature", label: "Temperature", min: 0, max: 1, step: 0.05, description: "LLM sampling temperature" },
  { key: "seed", label: "Seed", min: 0, max: 2 ** 31 - 1, step: 1, kind: "number", description: "Passed to the LLM where supported, for reproducibility" },
  { key: "graph_hops", label: "Graph hops", min: 0, max: 3, step: 1, description: "Graph neighbourhood around matched entities; 0 disables KG retrieval" },
  { key: "hybrid_weight_vector", label: "Vector weight", min: 0, max: 1, step: 0.05, description: "Vector vs graph weight in fusion (graph = 1 - value)" },
];

// build_graph is the one boolean; every other parameter is numeric.
export type ParamValues = Partial<Record<Exclude<ParamKey, "build_graph">, number>> & { build_graph?: boolean };
export type ParamErrors = Partial<Record<ParamKey, string>>;

/** Effective slider max, accounting for the cross-field rules. */
export function effectiveMax(meta: ParamMeta, values: ParamValues): number {
  if (meta.key === "chunk_overlap" && values.chunk_size != null)
    return Math.floor(values.chunk_size / 2);
  return meta.max;
}

function zodErrors(result: { success: boolean; error?: { issues: { path: PropertyKey[]; message: string }[] } }): ParamErrors {
  const out: ParamErrors = {};
  for (const issue of result.error?.issues ?? []) {
    const key = issue.path[0] as ParamKey;
    out[key] ??= issue.message;
  }
  return out;
}

export function validateIngest(v: ParamValues): ParamErrors {
  const pick = { chunk_size: v.chunk_size, chunk_overlap: v.chunk_overlap, dedup_threshold: v.dedup_threshold, build_graph: v.build_graph };
  const errors = zodErrors(IngestParamsSchema.safeParse(pick));
  if (!errors.chunk_overlap && v.chunk_size != null && v.chunk_overlap != null && v.chunk_overlap > Math.floor(v.chunk_size / 2))
    errors.chunk_overlap = `must be ≤ chunk size / 2 (${Math.floor(v.chunk_size / 2)})`;
  return errors;
}

export function validateQuery(v: ParamValues): ParamErrors {
  const keys = QUERY_PARAMS.map((p) => p.key);
  const pick = Object.fromEntries(keys.map((k) => [k, v[k]]));
  const errors = zodErrors(QueryParamsSchema.safeParse(pick));
  if (!errors.top_n && v.top_n != null && v.top_k != null && v.top_n > v.top_k)
    errors.top_n = `must be ≤ Top K (${v.top_k})`;
  return errors;
}

/** Values to send: only the keys of ``metas``. */
export function pickParams(metas: ParamMeta[], values: ParamValues): ParamValues {
  return Object.fromEntries(metas.map((m) => [m.key, values[m.key]]).filter(([, v]) => v != null));
}
