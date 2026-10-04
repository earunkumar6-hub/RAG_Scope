"use client";

import { useQueryClient } from "@tanstack/react-query";
import { Send } from "lucide-react";
import { useEffect, useState } from "react";

import { Header } from "@/components/layout/Header";
import { ParametersPanel, useRunParams } from "@/components/params/ParametersPanel";
import { PipelineStepper } from "@/components/pipeline/PipelineStepper";
import { RunHistory } from "@/components/pipeline/RunHistory";
import { AnswerPanel } from "@/components/query/AnswerPanel";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api, type FieldError } from "@/lib/api";
import { QUERY_PARAMS, validateQuery } from "@/lib/params";
import { QueryRequestSchema } from "@/lib/schemas.generated";
import { QUERY_STAGES, type StageId } from "@/lib/stages";
import { useRunId } from "@/lib/useRunId";
import { useStageEvents } from "@/lib/useStageEvents";

export function QueryView() {
  const [runId, setRunId] = useRunId();
  const { state: run, connection } = useStageEvents("query", runId);
  const { params, valid, errors: paramErrors } = useRunParams(QUERY_PARAMS, validateQuery);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<StageId | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [errors, setErrors] = useState<FieldError[]>([]);
  const queryClient = useQueryClient();

  const queryCheck = QueryRequestSchema.shape.query.safeParse(query.trim());
  const queryError = query.trim() && !queryCheck.success ? queryCheck.error.issues[0]?.message : null;

  useEffect(() => {
    if (run.done) queryClient.invalidateQueries({ queryKey: ["runs", "query"] });
  }, [run.done, queryClient]);

  async function submit() {
    setSubmitting(true);
    setErrors([]);
    try {
      const { run_id } = await api.query({ query: query.trim(), params });
      setSelected(null);
      setRunId(run_id);
    } catch (e) {
      setErrors(e instanceof ApiError ? e.details : [{ field: "", message: String(e) }]);
    } finally {
      setSubmitting(false);
    }
  }

  const running = !!runId && !run.done && connection !== "error";
  const canSubmit = queryCheck.success && valid && !submitting && !running;
  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        <Header title="Query" />
        <main className="flex-1 space-y-4 overflow-y-auto p-4">
          <div className="space-y-2">
            <Textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canSubmit) submit();
              }}
              placeholder="Ask a question about your documents… (Ctrl+Enter to run)"
              aria-invalid={!!queryError}
              className="min-h-20"
            />
            {queryError && <p className="text-xs text-destructive">query: {queryError}</p>}
            {errors.map((e) => (
              <p key={e.field + e.message} className="text-xs text-destructive">{e.field ? `${e.field}: ${e.message}` : e.message}</p>
            ))}
            <div className="flex items-center gap-2">
              <Button onClick={submit} disabled={!canSubmit}>
                <Send /> {running ? "Running…" : "Run query"}
              </Button>
              {Object.keys(paramErrors).length > 0 && (
              <span className="text-xs text-destructive">Fix the parameters first.</span>
            )}
              <div className="ml-auto flex items-center gap-2">
                {connection === "error" && <span className="text-xs text-destructive">Lost the event stream for this run.</span>}
                {run.done?.replayed && <span className="text-xs text-muted-foreground">Replayed from history</span>}
                <RunHistory kind="query" currentId={runId} onSelect={setRunId} />
              </div>
            </div>
          </div>
          {runId ? (
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <PipelineStepper stages={QUERY_STAGES} run={run} selected={selected} onSelect={setSelected} orientation="vertical" />
              <div className="order-first lg:sticky lg:top-0 lg:order-none lg:self-start">
                <AnswerPanel run={run} />
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Run a query to watch Q1–Q10 live. Click any stage for its full detail.</p>
          )}
        </main>
      </div>
      <ParametersPanel title="Query parameters" metas={QUERY_PARAMS} validate={validateQuery} />
    </div>
  );
}
