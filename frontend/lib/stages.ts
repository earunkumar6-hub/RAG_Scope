// Stage catalogue and the pure reducer that folds SSE messages into run state.
import type { StageEvent } from "@/lib/schemas.generated";

export type RunKind = "ingest" | "query";
export type StageId = StageEvent["stage_id"];
export type StageStatus = StageEvent["status"];

export type StageDef = { id: StageId; name: string; inputs: string; notBuilt?: string };

export const INGEST_STAGES: StageDef[] = [
  { id: "S1_upload", name: "Upload", inputs: "Uploaded files (1-10; .pdf .docx .md .txt, max 25 MB each)" },
  { id: "S2_parse", name: "Parse & Clean", inputs: "Raw file bytes from S1" },
  { id: "S3_tokenize", name: "Tokenize", inputs: "Clean text per document from S2" },
  { id: "S4_chunk", name: "Chunk", inputs: "Token stream per document from S3" },
  { id: "S5_embed", name: "Embed", inputs: "Chunk texts from S4" },
  { id: "S6_segregate", name: "Dedup & Cluster", inputs: "Chunk vectors from S5 plus the existing corpus in Chroma" },
  { id: "S7_vector_store", name: "Vector Store", inputs: "Chunks, vectors, duplicate links and cluster ids from S4-S6" },
  { id: "S8_kg_build", name: "Knowledge Graph", inputs: "New unique chunks from S7 (and the chunk ids S7 replaced)" },
];

export const QUERY_STAGES: StageDef[] = [
  { id: "Q1_validate", name: "Input Validation", inputs: "The QueryRequest envelope" },
  { id: "Q2_input_guardrail", name: "Input Guardrail", inputs: "Validated query text, PII already masked by the API" },
  { id: "Q3_query_embed", name: "Query Embedding", inputs: "Query text" },
  { id: "Q4_vector_retrieve", name: "Vector Retrieval", inputs: "Query vector from Q3 and filters" },
  { id: "Q5_graph_retrieve", name: "Graph Retrieval", inputs: "Query text, its entities (LLM) and the query vector from Q3" },
  { id: "Q6_fusion", name: "Hybrid Fusion", inputs: "Vector (Q4) and graph (Q5) candidates" },
  { id: "Q7_rerank", name: "Re-ranking", inputs: "Candidates that passed the threshold in Q4" },
  { id: "Q8_generate", name: "Generation", inputs: "Query plus the top-N re-ranked chunks from Q7" },
  { id: "Q9_output_guardrail", name: "Output Guardrail", inputs: "Streamed answer from Q8, its prompt and the top-N context" },
  { id: "Q10_eval", name: "Online Evaluation", inputs: "Query, context and answer" },
];

export type DonePayload = {
  status: string;
  error?: string;
  replayed?: boolean;
  answer?: string;
  original_answer?: string;
  blocked_by?: string[];
  citations?: Citation[];
  [key: string]: unknown;
};

export type Citation = {
  chunk_id: string;
  document_id?: string;
  filename?: string;
  page?: number;
  page_end?: number;
  similarity?: number;
  text?: string;
};

export type RunState = {
  stages: Partial<Record<StageId, StageEvent>>;
  tokens: string;
  /** Final query answer, sent after Q9 and before Q10 scores it (live runs only). */
  answer: DonePayload | null;
  done: DonePayload | null;
  lastSeq: number;
};

export const EMPTY_RUN: RunState = { stages: {}, tokens: "", answer: null, done: null, lastSeq: 0 };

export type SseMessage = { seq: number; event: "stage" | "token" | "answer" | "done"; data: unknown };

/** Fold one SSE message into the run state. Messages at or below ``lastSeq`` are ignored, so a
 * reconnect that overlaps already-seen messages cannot duplicate answer tokens. */
export function applyMessage(state: RunState, msg: SseMessage): RunState {
  if (msg.seq > 0 && msg.seq <= state.lastSeq) return state;
  const lastSeq = Math.max(state.lastSeq, msg.seq);
  switch (msg.event) {
    case "stage": {
      const ev = msg.data as StageEvent;
      return { ...state, lastSeq, stages: { ...state.stages, [ev.stage_id]: ev } };
    }
    case "token":
      return { ...state, lastSeq, tokens: state.tokens + (msg.data as { text: string }).text };
    case "answer":
      return { ...state, lastSeq, answer: msg.data as DonePayload };
    case "done":
      return { ...state, lastSeq, done: msg.data as DonePayload };
  }
}

/** True when the stage reported ``{"skipped": true}`` (not built yet or nothing to do). */
export function isSkipped(ev: StageEvent | undefined): boolean {
  return ev?.data?.skipped === true;
}

export function formatMs(ms: number | null | undefined): string {
  if (ms == null) return "";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}
