"use client";

import { Gauge } from "lucide-react";

import { ValueBar } from "@/components/pipeline/charts";
import { fmtMetric, ONLINE_METRICS } from "@/lib/eval";
import { formatMs, isSkipped, type RunState } from "@/lib/stages";

/* eslint-disable @typescript-eslint/no-explicit-any -- stage payloads are free-form JSON */

/** Q10 online-evaluation gauges under the answer; they fill in after the answer is shown. */
export function EvalGauges({ run }: { run: RunState }) {
  const q10 = run.stages.Q10_eval;
  if (!q10 || q10.status === "pending" || run.done?.status === "blocked") return null;
  if (isSkipped(q10)) {
    return <p className="text-xs text-muted-foreground">Online evaluation is off (Settings).</p>;
  }
  const data = (q10.data ?? {}) as Record<string, any>;
  const metrics = (data.metrics ?? {}) as Record<string, { value: number | null; note?: string }>;
  const total = data.tokens?.total;
  const scoring = q10.status === "running";
  return (
    <section className="space-y-2" aria-label="Online evaluation">
      <h4 className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
        <Gauge aria-hidden className="size-3.5" /> Online evaluation
        {scoring && <span className="font-normal normal-case">scoring…</span>}
      </h4>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {ONLINE_METRICS.map((m) => {
          const metric = metrics[m.id];
          const value = metric?.value;
          return (
            <div key={m.id} className="space-y-1 rounded-md border p-2" title={m.help}>
              <div className="text-[11px] text-muted-foreground">{m.label}</div>
              <div className="text-lg font-semibold tabular-nums">{scoring ? "…" : fmtMetric(value)}</div>
              {value != null ? <ValueBar value={value} /> : <div className="h-1.5 rounded-full bg-muted" />}
              {!scoring && value == null && metric?.note && (
                <p className="text-[11px] text-muted-foreground">{metric.note}</p>
              )}
            </div>
          );
        })}
      </div>
      {total && (
        <p className="text-[11px] text-muted-foreground">
          {total.calls} LLM call(s) · {(total.prompt_tokens + total.completion_tokens).toLocaleString()} tokens
          {total.cache_hits > 0 && ` · ${total.cache_hits} cached (0 tokens)`} · answer ready in{" "}
          {formatMs(data.latency_ms?.answer_ready)}
        </p>
      )}
    </section>
  );
}
