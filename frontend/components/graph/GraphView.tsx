"use client";

import dynamic from "next/dynamic";
import { useTheme } from "next-themes";
import { useEffect, useMemo, useRef, useState } from "react";

import type { CanvasColors } from "@/components/graph/GraphCanvas";

// Canvas + d3-force need the browser.
const GraphCanvas = dynamic(() => import("@/components/graph/GraphCanvas"), {
  ssr: false,
  loading: () => <p className="p-4 text-xs text-muted-foreground">Loading graph…</p>,
});

export type GraphVertex = {
  id: string;
  name: string;
  type: string;
  description?: string;
  aliases?: string[];
  chunk_ids?: string[];
  degree?: number;
  hop?: number;
  matched?: boolean;
};
export type GraphEdge = { source: string; target: string; relation: string; chunk_ids?: string[] };

// Fixed type -> categorical slot (colour follows the entity type, never its rank or count).
export const TYPE_SLOTS = ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "TECHNOLOGY", "CONCEPT", "EVENT", "OTHER"];

function readColors(): CanvasColors {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    types: Object.fromEntries(
      TYPE_SLOTS.map((t, i) => [t, t === "OTHER" ? v("--series-other") : v(`--series-${i + 1}`)]),
    ),
    ink: v("--foreground"),
    muted: v("--muted-foreground"),
    edge: v("--muted-foreground"),
  };
}

export function typeColor(type: string): string {
  const slot = TYPE_SLOTS.indexOf(type);
  return slot < 0 || type === "OTHER" ? "var(--series-other)" : `var(--series-${slot + 1})`;
}

export function TypeLegend({ types }: { types: string[] }) {
  return (
    <ul className="flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
      {TYPE_SLOTS.filter((t) => types.includes(t)).map((t) => (
        <li key={t} className="flex items-center gap-1">
          <span className="size-2.5 rounded-full" style={{ background: typeColor(t) }} />
          {t.toLowerCase()}
        </li>
      ))}
    </ul>
  );
}

/** Responsive force graph with a type legend. ``matched`` vertices are ringed. */
export function GraphView({
  vertices,
  edges,
  height = 320,
  selected,
  onSelect,
}: {
  vertices: GraphVertex[];
  edges: GraphEdge[];
  height?: number;
  selected?: string | null;
  onSelect?: (v: GraphVertex) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const { resolvedTheme } = useTheme();
  // Re-read the palette when the theme flips; canvas cannot use CSS variables directly.
  const colors = useMemo(() => (typeof window === "undefined" ? null : readColors()), [resolvedTheme]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const types = [...new Set(vertices.map((v) => v.type))];
  return (
    <div className="space-y-1.5">
      <div ref={box} className="overflow-hidden rounded-md border bg-card" style={{ height }}>
        {vertices.length === 0 ? (
          <p className="p-4 text-xs text-muted-foreground">No vertices to show.</p>
        ) : (
          width > 0 &&
          colors && (
            <GraphCanvas
              vertices={vertices}
              edges={edges}
              width={width}
              height={height}
              colors={colors}
              selected={selected}
              onSelect={onSelect}
            />
          )
        )}
      </div>
      <TypeLegend types={types} />
    </div>
  );
}
