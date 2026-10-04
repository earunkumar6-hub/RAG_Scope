"use client";

import { useQueryClient } from "@tanstack/react-query";
import { Play } from "lucide-react";
import { useEffect, useState } from "react";

import { UploadZone, fileProblems } from "@/components/ingest/UploadZone";
import { Header } from "@/components/layout/Header";
import { ParametersPanel, useRunParams } from "@/components/params/ParametersPanel";
import { PipelineStepper } from "@/components/pipeline/PipelineStepper";
import { RunHistory } from "@/components/pipeline/RunHistory";
import { Button } from "@/components/ui/button";
import { ApiError, type FieldError } from "@/lib/api";
import { api } from "@/lib/api";
import { INGEST_PARAMS, validateIngest } from "@/lib/params";
import { INGEST_STAGES, type StageId } from "@/lib/stages";
import { useRunId } from "@/lib/useRunId";
import { useStageEvents } from "@/lib/useStageEvents";

export function IngestView() {
  const [runId, setRunId] = useRunId();
  const { state: run, connection } = useStageEvents("ingest", runId);
  const { params, valid, errors: paramErrors } = useRunParams(INGEST_PARAMS, validateIngest);
  const [files, setFiles] = useState<File[]>([]);
  const [selected, setSelected] = useState<StageId | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [errors, setErrors] = useState<FieldError[]>([]);
  const queryClient = useQueryClient();
  const problems = fileProblems(files);

  useEffect(() => {
    if (run.done) queryClient.invalidateQueries({ queryKey: ["runs", "ingest"] });
  }, [run.done, queryClient]);

  async function start() {
    setSubmitting(true);
    setErrors([]);
    try {
      const { job_id } = await api.ingest(files, params);
      setFiles([]);
      setSelected(null);
      setRunId(job_id);
    } catch (e) {
      setErrors(e instanceof ApiError ? e.details : [{ field: "", message: String(e) }]);
    } finally {
      setSubmitting(false);
    }
  }

  const running = !!runId && !run.done && connection !== "error";
  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        <Header title="Ingest" />
        <main className="flex-1 space-y-4 overflow-y-auto p-4">
          <UploadZone files={files} onChange={setFiles} />
          {[...problems, ...errors.map((e) => (e.field ? `${e.field}: ${e.message}` : e.message))].map((p) => (
            <p key={p} className="text-xs text-destructive">{p}</p>
          ))}
          <div className="flex items-center gap-2">
            <Button onClick={start} disabled={!files.length || problems.length > 0 || !valid || submitting || running}>
              <Play /> {running ? "Ingesting…" : "Ingest"}
            </Button>
            {Object.keys(paramErrors).length > 0 && (
              <span className="text-xs text-destructive">Fix the parameters first.</span>
            )}
            <div className="ml-auto flex items-center gap-2">
              {connection === "error" && <span className="text-xs text-destructive">Lost the event stream for this run.</span>}
              {run.done && (
                <span className="text-xs text-muted-foreground">
                  Finished: {run.done.status}
                  {run.done.replayed && " (replayed from history)"}
                  {run.done.error && ` - ${run.done.error}`}
                </span>
              )}
              <RunHistory kind="ingest" currentId={runId} onSelect={setRunId} />
            </div>
          </div>
          {runId ? (
            <PipelineStepper stages={INGEST_STAGES} run={run} selected={selected} onSelect={setSelected} orientation="horizontal" />
          ) : (
            <p className="text-sm text-muted-foreground">Upload documents to watch S1–S7 run live. Click any stage card for its full detail.</p>
          )}
        </main>
      </div>
      <ParametersPanel title="Ingest parameters" metas={INGEST_PARAMS} validate={validateIngest} />
    </div>
  );
}
