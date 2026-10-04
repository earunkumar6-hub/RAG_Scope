"use client";

import { cn } from "cn";

import { td } from "@/components/pipeline/Table";
import type { EvalCellSummary } from "@/lib/api";
import { cellLabel, fmtMetric, METRICS } from "@/lib/eval";
import { formatMs } from "@/lib/stages";

/** One row per grid cell with the mean of every metric; the best mean per metric is bold.
 * Clicking a row selects that cell for the per-question drill-down. */
export function MetricsTable({
  summary,
  varying,
  selected,
  onSelect,
}: {
  summary: EvalCellSummary[];
  varying: string[];
  selected: number;
  onSelect: (cell: number) => void;
}) {
  const best = Object.fromEntries(
    METRICS.map((m) => [m.id, Math.max(...summary.map((s) => s.metrics[m.id]?.mean ?? -1))]),
  );
  return (
    <div className="overflow-x-auto rounded border">
      <table className="w-full text-xs">
        <thead className="bg-muted text-left text-muted-foreground">
          <tr>
            <th className="px-2 py-1 font-medium">Cell</th>
            {METRICS.map((m) => (
              <th key={m.id} className="px-2 py-1 font-medium" title={m.help}>{m.label}</th>
            ))}
            <th className="px-2 py-1 font-medium">Mean latency</th>
            <th className="px-2 py-1 font-medium">Tokens</th>
          </tr>
        </thead>
        <tbody className="divide-y">
          {summary.map((s) => (
            <tr
              key={s.cell}
              onClick={() => onSelect(s.cell)}
              aria-selected={selected === s.cell}
              className={cn("cursor-pointer hover:bg-muted/50", selected === s.cell && "bg-muted")}
            >
              <td className={cn(td, "font-medium whitespace-nowrap")}>
                {cellLabel(s.params, varying)}
                {s.partial && (
                  <span className="ml-1 font-normal text-muted-foreground" title="The run stopped before this cell finished">
                    (partial, {s.questions} q)
                  </span>
                )}
              </td>
              {METRICS.map((m) => {
                const v = s.metrics[m.id];
                const isBest = summary.length > 1 && v?.mean != null && v.mean === best[m.id];
                return (
                  <td key={m.id} className={cn(td, "tabular-nums", isBest && "font-semibold")} title={v ? `${v.n} value(s)` : undefined}>
                    {fmtMetric(v?.mean)}
                  </td>
                );
              })}
              <td className={cn(td, "tabular-nums")}>{formatMs(s.mean_latency_ms)}</td>
              <td className={cn(td, "tabular-nums")}>{s.total_tokens.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
