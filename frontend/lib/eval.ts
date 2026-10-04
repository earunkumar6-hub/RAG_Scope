// Evaluation metric labels and the pure helpers behind the Eval page's grid builder.

export type MetricName =
  | "faithfulness"
  | "answer_relevancy"
  | "context_precision"
  | "context_recall"
  | "answer_correctness"
  | "hit_rate"
  | "mrr";

export const METRICS: { id: MetricName; label: string; help: string }[] = [
  { id: "faithfulness", label: "Faithfulness", help: "Share of answer sentences the context supports (LLM judge)" },
  { id: "answer_relevancy", label: "Answer relevancy", help: "Cosine between question and answer embeddings" },
  { id: "context_precision", label: "Context precision", help: "Share of context chunks relevant to the reference answer (LLM judge)" },
  { id: "context_recall", label: "Context recall", help: "Share of reference-answer statements found in the context (LLM judge)" },
  { id: "answer_correctness", label: "Answer correctness", help: "0.75 × factual F1 vs the reference + 0.25 × similarity. Extra detail the reference doesn't mention is neutral; only contradictions count against the answer (unlike RAGAS)" },
  { id: "hit_rate", label: "Hit rate@top_n", help: "A relevant chunk is in the context (questions with relevant chunks only)" },
  { id: "mrr", label: "MRR", help: "1 / rank of the first relevant chunk in the context" },
];

// Online (Q10) precision has no ground truth: it judges relevance to the query.
export const ONLINE_METRICS: { id: MetricName; label: string; help: string }[] = [
  { id: "faithfulness", label: "Faithfulness", help: METRICS[0].help },
  { id: "answer_relevancy", label: "Answer relevancy", help: METRICS[1].help },
  { id: "context_precision", label: "Context precision (vs query)", help: "Share of context chunks relevant to the question (LLM judge)" },
];

export const fmtMetric = (v: number | null | undefined) => (v == null ? "–" : v.toFixed(2));

export type GridInput = Record<string, string>;
export type GridParse = { grid: Record<string, number[]>; errors: Record<string, string>; cells: number };

/** Parse comma-separated values per parameter; empty inputs are left at their default. */
export function parseGrid(input: GridInput, maxValues: number): GridParse {
  const grid: Record<string, number[]> = {};
  const errors: Record<string, string> = {};
  for (const [key, raw] of Object.entries(input)) {
    const parts = raw.split(",").map((p) => p.trim()).filter(Boolean);
    if (!parts.length) continue;
    const values = parts.map(Number);
    if (values.some((v) => !Number.isFinite(v))) errors[key] = "numbers separated by commas";
    else if (new Set(values).size !== values.length) errors[key] = "values must be distinct";
    else if (values.length > maxValues) errors[key] = `at most ${maxValues} values`;
    else grid[key] = values;
  }
  const cells = Object.values(grid).reduce((n, v) => n * v.length, 1);
  return { grid, errors, cells };
}

/** Short label for a grid cell: only the parameters that vary across the run's cells. */
export function cellLabel(params: Record<string, unknown>, varying: string[]): string {
  return varying.length ? varying.map((k) => `${k}=${params[k]}`).join(", ") : "defaults";
}

export function varyingKeys(cells: Record<string, unknown>[]): string[] {
  if (cells.length < 2) return [];
  return Object.keys(cells[0]).filter((k) => cells.some((c) => c[k] !== cells[0][k]));
}
