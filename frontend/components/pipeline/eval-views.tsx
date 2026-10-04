"use client";

// Q10 online-evaluation stage detail: metrics with their notes, judge verdicts, per-stage
// latency and LLM token usage.
import { ValueBar } from "@/components/pipeline/charts";
import { Table, td } from "@/components/pipeline/Table";
import { fmtMetric, ONLINE_METRICS } from "@/lib/eval";
import { formatMs, QUERY_STAGES } from "@/lib/stages";

/* eslint-disable @typescript-eslint/no-explicit-any -- stage payloads are free-form JSON */
type Data = Record<string, any>;

const stageName = (id: string) => QUERY_STAGES.find((s) => s.id === id)?.name ?? id;

export function Q10View({ data }: { data: Data }) {
  const metrics: Data = data.metrics ?? {};
  const stages: Record<string, number> = data.latency_ms?.stages ?? {};
  const slowest = Math.max(1, ...Object.values(stages));
  const tokens: Record<string, Data> = data.tokens?.by_stage ?? {};
  return (
    <div className="space-y-3 text-xs">
      <Table head={["Metric", "Value", "", "Note"]}>
        {ONLINE_METRICS.map((m) => {
          const metric: Data = metrics[m.id] ?? {};
          return (
            <tr key={m.id} title={m.help}>
              <td className={td}>{m.label}</td>
              <td className={`${td} tabular-nums`}>{fmtMetric(metric.value)}</td>
              <td className={`${td} w-32`}>{metric.value != null && <ValueBar value={metric.value} />}</td>
              <td className={`${td} text-muted-foreground`}>
                {metric.note || (metric.source === "q9" ? "reused Q9's groundedness verdicts" : "")}
              </td>
            </tr>
          );
        })}
      </Table>
      {metrics.context_precision?.verdicts?.length > 0 && (
        <details>
          <summary className="cursor-pointer text-muted-foreground">Precision judge (relevance to the query)</summary>
          <ul className="mt-1 space-y-0.5">
            {metrics.context_precision.verdicts.map((v: Data) => (
              <li key={v.chunk_id} className="flex gap-2">
                <span className="w-16 shrink-0">{v.relevant ? "relevant" : "not relevant"}</span>
                <span className="font-mono">{v.chunk_id}</span>
                <span className="text-muted-foreground">{v.reason}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      <div className="space-y-1">
        <h4 className="font-medium">Latency per stage · answer ready in {formatMs(data.latency_ms?.answer_ready)}</h4>
        <Table head={["Stage", "Time", ""]}>
          {Object.entries(stages).map(([id, ms]) => (
            <tr key={id}>
              <td className={td}>{stageName(id)}</td>
              <td className={`${td} tabular-nums`}>{formatMs(ms)}</td>
              <td className={`${td} w-40`}><ValueBar value={ms / slowest} /></td>
            </tr>
          ))}
        </Table>
      </div>
      <div className="space-y-1">
        <h4 className="font-medium">LLM token usage</h4>
        <Table head={["Stage", "Calls", "Cached", "Prompt", "Completion"]}>
          {Object.entries(tokens).map(([id, t]) => (
            <tr key={id}>
              <td className={td}>{stageName(id)}</td>
              <td className={`${td} tabular-nums`}>{t.calls}</td>
              <td className={`${td} tabular-nums`}>{t.cache_hits}</td>
              <td className={`${td} tabular-nums`}>{t.prompt_tokens.toLocaleString()}</td>
              <td className={`${td} tabular-nums`}>{t.completion_tokens.toLocaleString()}</td>
            </tr>
          ))}
        </Table>
        <p className="text-muted-foreground">Cached calls reuse a stored result and cost 0 tokens.</p>
      </div>
    </div>
  );
}
