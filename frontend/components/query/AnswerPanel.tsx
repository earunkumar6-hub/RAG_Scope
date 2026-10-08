"use client";

import { useQuery } from "@tanstack/react-query";
import { Ban, FileSearch, ShieldCheck } from "lucide-react";
import { useState } from "react";

import { AnswerDiff } from "@/components/pipeline/guard-views";
import { ChunkDialog } from "@/components/query/ChunkDialog";
import { EvalGauges } from "@/components/query/EvalGauges";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { api } from "@/lib/api";
import { splitAnswer } from "@/lib/citations";
import type { Citation, RunState } from "@/lib/stages";

/* eslint-disable @typescript-eslint/no-explicit-any -- stage payloads are free-form JSON */

/** Streaming answer with clickable [chunk_id] citations, plus the prompt/context drawer. */
export function AnswerPanel({ run }: { run: RunState }) {
  const [openChunk, setOpenChunk] = useState<string | null>(null);
  const [drawer, setDrawer] = useState(false);
  const [showDiff, setShowDiff] = useState(false);
  const q8 = run.stages.Q8_generate;
  const q8data = (q8?.data ?? {}) as Record<string, any>;
  const q9data = (run.stages.Q9_output_guardrail?.data ?? {}) as Record<string, any>;
  // While streaming: the live tokens. Once Q9 / the run is done: the final (possibly corrected)
  // answer, which arrives as ``answer`` before Q10 scores it. Runs replayed from the database
  // have no tokens, only stored answers.
  const final = run.done ?? run.answer;
  const answer = final?.answer ?? q9data.final_answer ?? (run.tokens || q8data.answer || "");
  const original: string | undefined = final?.original_answer ?? (q9data.modified ? q9data.original_answer : undefined);
  const blocked = run.done?.status === "blocked";
  const citations: Citation[] = final?.citations ?? q8data.citations ?? [];
  // Q8's context list arrives with its terminal event; while streaming use Q7's kept chunks.
  const q7kept = ((run.stages.Q7_rerank?.data?.ranking ?? []) as any[]).filter((r) => r.kept).map((r) => r.chunk_id);
  const contextIds: string[] = q8data.context_chunk_ids ?? q7kept;
  const known = new Map<string, Citation>(citations.map((c) => [c.chunk_id, c]));
  const valid = new Set([...contextIds, ...known.keys()]);
  const streaming = q8?.status === "running";
  const error = run.done?.status === "error" ? run.done.error : null;
  const prompt: { role: string; content: string }[] | undefined = q8data.prompt;
  const index = (id: string) => [...valid].indexOf(id) + 1;
  // Settings > Display: off shows sources as file and page only (also for replayed runs).
  // Until the config loads, citations stay closed.
  const config = useQuery({ queryKey: ["config"], queryFn: api.config });
  const showText = config.data?.display.citation_chunk_text.enabled === true;
  const where = (c?: Citation) => (c ? `${c.filename} p. ${c.page}` : "");

  return (
    <div className="space-y-3 rounded-lg border bg-card p-4">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold">Answer</h3>
        {streaming && <Badge variant="secondary">streaming…</Badge>}
        <Button className="ml-auto" variant="outline" size="sm" disabled={!prompt} onClick={() => setDrawer(true)}>
          <FileSearch /> Prompt & context
        </Button>
      </div>
      {!answer && !error && (
        <p className="text-sm text-muted-foreground">{run.done ? "No answer was produced." : "The answer appears here as it is generated."}</p>
      )}
      {blocked && (
        <div className="flex gap-2 rounded-md border border-red-500/40 bg-red-500/5 p-3 text-sm" role="alert">
          <Ban aria-hidden className="mt-0.5 size-4 shrink-0 text-red-600 dark:text-red-500" />
          <p>{answer}</p>
        </div>
      )}
      {original && !blocked && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <ShieldCheck aria-hidden className="size-3.5 text-amber-600 dark:text-amber-500" />
          <span>Corrected by the output guardrails.</span>
          <Button size="xs" variant="ghost" onClick={() => setShowDiff((v) => !v)}>
            {showDiff ? "Hide" : "Show"} original vs final
          </Button>
        </div>
      )}
      {original && showDiff && !blocked && <AnswerDiff before={original} after={answer} />}
      {answer && !blocked && (
        <p className="text-sm leading-relaxed whitespace-pre-wrap" data-testid="answer">
          {splitAnswer(answer, valid).map((seg, i) =>
            seg.type === "text" ? (
              <span key={i}>{seg.text}</span>
            ) : !showText ? (
              <span
                key={i}
                title={where(known.get(seg.chunkId))}
                className="mx-0.5 rounded bg-primary/10 px-1 align-super text-[10px] font-medium text-primary"
              >
                {index(seg.chunkId)}
              </span>
            ) : (
              <button
                key={i}
                type="button"
                title={seg.chunkId}
                onClick={() => setOpenChunk(seg.chunkId)}
                className="mx-0.5 rounded bg-primary/10 px-1 align-super text-[10px] font-medium text-primary hover:bg-primary/20"
              >
                {index(seg.chunkId)}
              </button>
            ),
          )}
          {streaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-foreground/60 align-middle" />}
        </p>
      )}
      {error && <p className="text-sm text-destructive">Run failed: {error}</p>}
      {citations.length > 0 && (
        <div className="space-y-1">
          <h4 className="text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">Sources</h4>
          <ol className="space-y-1">
            {citations.map((c) => (
              <li key={c.chunk_id}>
                {!showText ? (
                  <div className="flex gap-2 px-1 py-0.5 text-xs">
                    <span className="font-medium text-primary">{index(c.chunk_id)}</span>
                    <span>{where(c)}</span>
                  </div>
                ) : (
                <button
                  type="button"
                  onClick={() => setOpenChunk(c.chunk_id)}
                  className="flex w-full gap-2 rounded px-1 py-0.5 text-left text-xs hover:bg-muted"
                >
                  <span className="font-medium text-primary">{index(c.chunk_id)}</span>
                  <span>
                    {c.filename} p. {c.page}
                    <span className="ml-1 font-mono text-muted-foreground">{c.chunk_id}</span>
                  </span>
                </button>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}
      <EvalGauges run={run} />
      <ChunkDialog chunkId={showText ? openChunk : null} filename={known.get(openChunk ?? "")?.filename} onClose={() => setOpenChunk(null)} />
      <Sheet open={drawer} onOpenChange={setDrawer}>
        <SheetContent side="right" className="w-full sm:max-w-2xl">
          <SheetHeader>
            <SheetTitle>Prompt & context sent to the LLM</SheetTitle>
            <SheetDescription>
              {contextIds.length} context chunk(s) · {q8data.model ?? ""} · {q8data.prompt_tokens ?? "?"} prompt tokens
            </SheetDescription>
          </SheetHeader>
          <div className="flex-1 space-y-3 overflow-y-auto px-4 pb-4">
            {prompt?.map((m, i) => (
              <div key={i} className="space-y-1">
                <Badge variant="outline">{m.role}</Badge>
                <pre className="rounded bg-muted p-2 font-mono text-[11px] whitespace-pre-wrap">{m.content}</pre>
              </div>
            ))}
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}
