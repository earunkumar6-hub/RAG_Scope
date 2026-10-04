"use client";

// Compare up to 4 grid cells: grouped bars per metric and a radar of the same means. Each
// selected cell keeps its color slot while selected (color follows the cell, not its position).
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AXIS, TOOLTIP } from "@/components/pipeline/charts";
import type { EvalCellSummary } from "@/lib/api";
import { cellLabel, METRICS } from "@/lib/eval";

export const MAX_COMPARED = 4;

export type Compared = { cell: number; slot: number }[];

/** Toggle ``cell``; a newly selected cell takes the lowest free color slot. */
export function toggleCompared(compared: Compared, cell: number): Compared {
  if (compared.some((c) => c.cell === cell)) return compared.filter((c) => c.cell !== cell);
  if (compared.length >= MAX_COMPARED) return compared;
  const used = new Set(compared.map((c) => c.slot));
  const slot = [1, 2, 3, 4].find((s) => !used.has(s)) ?? 1;
  return [...compared, { cell, slot }];
}

export function MetricsChart({
  summary,
  compared,
  varying,
}: {
  summary: EvalCellSummary[];
  compared: Compared;
  varying: string[];
}) {
  const series = compared
    .map((c) => ({ ...c, s: summary.find((s) => s.cell === c.cell) }))
    .filter((c): c is { cell: number; slot: number; s: EvalCellSummary } => !!c.s)
    .map((c) => ({ key: `c${c.cell}`, name: cellLabel(c.s.params, varying), color: `var(--series-${c.slot})`, s: c.s }));
  // Only metrics every compared cell has a value for (e.g. hit rate needs relevant chunks).
  const metrics = METRICS.filter((m) => series.length && series.every((c) => c.s.metrics[m.id]?.mean != null));
  const data = metrics.map((m) => ({
    metric: m.label,
    ...Object.fromEntries(series.map((c) => [c.key, c.s.metrics[m.id].mean])),
  }));
  if (!series.length) return <p className="text-xs text-muted-foreground">Pick cells to compare.</p>;
  if (!data.length) return <p className="text-xs text-muted-foreground">No metric has values for every selected cell.</p>;
  const fmt = (v: unknown) => (typeof v === "number" ? v.toFixed(2) : String(v));
  // Legend text stays in ink; the swatch beside it carries the series color.
  const legend = { wrapperStyle: { fontSize: 11 }, formatter: (v: string) => <span style={{ color: "var(--foreground)" }}>{v}</span> };
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <figure className="space-y-1">
        <figcaption className="text-xs font-medium">Mean metric per cell</figcaption>
        <div className="h-72">
          <ResponsiveContainer>
            <BarChart data={data} barGap={2} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid vertical={false} stroke="var(--border)" strokeOpacity={0.6} />
              <XAxis dataKey="metric" tick={AXIS} tickLine={false} axisLine={{ stroke: "var(--border)" }} interval={0} angle={-20} textAnchor="end" height={64} />
              <YAxis domain={[0, 1]} tick={AXIS} tickLine={false} axisLine={false} />
              <Tooltip {...TOOLTIP} formatter={(v, name) => [fmt(v), name]} />
              <Legend {...legend} />
              {series.map((c) => (
                <Bar key={c.key} dataKey={c.key} name={c.name} fill={c.color} radius={[4, 4, 0, 0]} isAnimationActive={false} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </figure>
      <figure className="space-y-1">
        <figcaption className="text-xs font-medium">Metric profile</figcaption>
        <div className="h-72">
          <ResponsiveContainer>
            <RadarChart data={data} outerRadius="70%">
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey="metric" tick={AXIS} />
              <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
              <Tooltip {...TOOLTIP} formatter={(v, name) => [fmt(v), name]} />
              <Legend {...legend} />
              {series.map((c) => (
                <Radar key={c.key} dataKey={c.key} name={c.name} stroke={c.color} strokeWidth={2} fill={c.color} fillOpacity={0.12} dot={{ r: 3 }} isAnimationActive={false} />
              ))}
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </figure>
    </div>
  );
}
