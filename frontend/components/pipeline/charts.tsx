"use client";

// Recharts views for stage outputs. Colors come from the --series-* CSS variables (validated
// categorical palette, light and dark steps); text stays in foreground/muted ink.
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const AXIS = { fontSize: 11, fill: "var(--muted-foreground)" };
export const TOOLTIP = {
  contentStyle: {
    background: "var(--popover)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    fontSize: 12,
    color: "var(--popover-foreground)",
  },
  itemStyle: { color: "var(--popover-foreground)" }, // values in ink, not series colour
  cursor: { fill: "var(--muted)", opacity: 0.5 },
};

export type HistogramBin = { from: number; to: number; count: number };

/** Chunk token-count histogram (single series: no legend, the title names it). */
export function TokenHistogram({ bins }: { bins: HistogramBin[] }) {
  const data = bins.map((b) => ({ ...b, range: `${b.from}–${b.to}` }));
  return (
    <figure className="space-y-1">
      <figcaption className="text-xs font-medium">Chunk size distribution (tokens)</figcaption>
      <div className="h-48">
        <ResponsiveContainer>
          <BarChart data={data} barCategoryGap={2} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid vertical={false} stroke="var(--border)" strokeOpacity={0.6} />
            <XAxis dataKey="range" tick={AXIS} tickLine={false} axisLine={{ stroke: "var(--border)" }} interval="preserveStartEnd" />
            <YAxis allowDecimals={false} tick={AXIS} tickLine={false} axisLine={false} />
            <Tooltip {...TOOLTIP} formatter={(v) => [v, "chunks"]} labelFormatter={(l) => `${l} tokens`} />
            <Bar dataKey="count" fill="var(--series-1)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export type ScatterPoint = { chunk_id: string; x: number; y: number; cluster?: number | null };

// Fixed slot order; each slot also gets its own marker shape so identity is never color alone
// (a scatter puts every pair of hues side by side). Clusters past the 8th fold into "Other".
const SLOTS = 8;
const SHAPES = ["circle", "square", "triangle", "diamond", "star", "cross", "wye", "circle"] as const;

/** PCA projection of chunk embeddings, colored and shaped by topic cluster. */
export function EmbeddingScatter({
  points,
  labels,
}: {
  points: ScatterPoint[];
  labels: Record<number, string>;
}) {
  const clusterIds = [...new Set(points.map((p) => p.cluster ?? -1))].sort((a, b) => a - b);
  const groups = new Map<string, { name: string; slot: number; points: ScatterPoint[] }>();
  clusterIds.forEach((cid, i) => {
    const folded = cid < 0 || i >= SLOTS;
    const key = folded ? "other" : String(cid);
    if (!groups.has(key))
      groups.set(key, {
        name: folded ? (cid < 0 ? "Unassigned" : "Other clusters") : (labels[cid] ?? `Cluster ${cid}`),
        slot: folded ? -1 : i,
        points: [],
      });
    groups.get(key)!.points.push(...points.filter((p) => (p.cluster ?? -1) === cid));
  });
  return (
    <figure className="space-y-1">
      <figcaption className="text-xs font-medium">Chunk embeddings (PCA, 2D) by topic cluster</figcaption>
      <div className="h-72">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
            <CartesianGrid stroke="var(--border)" strokeOpacity={0.6} />
            <XAxis type="number" dataKey="x" name="PC1" tick={AXIS} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
            <YAxis type="number" dataKey="y" name="PC2" tick={AXIS} tickLine={false} axisLine={false} />
            <Tooltip
              {...TOOLTIP}
              cursor={{ strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                const p = active && payload?.[0]?.payload as ScatterPoint | undefined;
                if (!p) return null;
                return (
                  <div style={TOOLTIP.contentStyle} className="px-2 py-1">
                    <div className="font-mono">{p.chunk_id}</div>
                    <div className="text-muted-foreground">
                      {p.cluster != null && p.cluster >= 0 ? (labels[p.cluster] ?? `Cluster ${p.cluster}`) : "Unassigned"}
                    </div>
                  </div>
                );
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: 11 }}
              // label text stays in ink; the coloured marker carries identity
              formatter={(value) => <span style={{ color: "var(--foreground)" }}>{value}</span>}
            />
            {[...groups.values()].map((g) => (
              <Scatter
                key={g.name}
                name={g.name}
                data={g.points}
                fill={g.slot < 0 ? "var(--series-other)" : `var(--series-${g.slot + 1})`}
                shape={g.slot < 0 ? "circle" : SHAPES[g.slot]}
                legendType={g.slot < 0 ? "circle" : SHAPES[g.slot]}
                stroke="var(--card)"
                strokeWidth={1}
                isAnimationActive={false}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

/** Thin horizontal bar for a 0..1 value with an optional threshold marker. */
export function ValueBar({ value, threshold, muted }: { value: number; threshold?: number; muted?: boolean }) {
  return (
    <div className="relative h-1.5 w-full min-w-16 rounded-full bg-muted">
      <div
        className="h-full rounded-full"
        style={{
          width: `${Math.max(0, Math.min(1, value)) * 100}%`,
          background: muted ? "var(--series-other)" : "var(--series-1)",
        }}
      />
      {threshold != null && (
        <div
          className="absolute -top-0.5 h-2.5 w-0.5 bg-foreground/70"
          style={{ left: `${threshold * 100}%` }}
          title={`threshold ${threshold}`}
        />
      )}
    </div>
  );
}
