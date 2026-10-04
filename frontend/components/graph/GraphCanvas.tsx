"use client";

// Canvas force graph (react-force-graph-2d). Loaded only in the browser via GraphView.
import { useEffect, useMemo, useRef } from "react";
import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from "react-force-graph-2d";

import type { GraphEdge, GraphVertex } from "@/components/graph/GraphView";

type Node = NodeObject<GraphVertex>;
type Link = LinkObject<GraphVertex, GraphEdge>;

const MAX_ZOOM = 3;

export type CanvasColors = { types: Record<string, string>; ink: string; muted: string; edge: string };

export default function GraphCanvas({
  vertices,
  edges,
  width,
  height,
  colors,
  selected,
  onSelect,
}: {
  vertices: GraphVertex[];
  edges: GraphEdge[];
  width: number;
  height: number;
  colors: CanvasColors;
  selected?: string | null;
  onSelect?: (v: GraphVertex) => void;
}) {
  const ref = useRef<ForceGraphMethods<Node, Link> | undefined>(undefined);
  // New objects only when the data changes, so the simulation is not restarted on every render.
  const data = useMemo(
    () => ({
      nodes: vertices.map((v) => ({ ...v })),
      links: edges.map((e) => ({ ...e })),
    }),
    [vertices, edges],
  );
  const showAllLabels = vertices.length <= 40;
  // Spread nodes enough for their labels (d3 defaults pack small graphs tightly).
  useEffect(() => {
    const fg = ref.current;
    if (!fg) return;
    fg.d3Force("charge")?.strength(-180);
    fg.d3Force("link")?.distance(55);
    fg.d3ReheatSimulation();
  }, [data]);

  // Fit whenever the layout settles (initial run, re-heat, drag); cap the zoom so tiny graphs
  // stay readable.
  const fit = () => {
    const fg = ref.current;
    if (!fg) return;
    fg.zoomToFit(300, 40);
    setTimeout(() => {
      if (fg.zoom() > MAX_ZOOM) fg.zoom(MAX_ZOOM, 200);
    }, 320);
  };

  return (
    <ForceGraph2D<GraphVertex, GraphEdge>
      ref={ref}
      graphData={data}
      width={width}
      height={height}
      backgroundColor="rgba(0,0,0,0)"
      cooldownTicks={120}
      onEngineStop={fit}
      nodeLabel={(n) => `${n.name} (${n.type})${n.description ? ` - ${n.description}` : ""}`}
      linkLabel={(l) => l.relation}
      linkColor={() => colors.edge}
      linkWidth={1.2}
      linkDirectionalArrowLength={4}
      linkDirectionalArrowRelPos={1}
      onNodeClick={(n) => onSelect?.(n)}
      nodeCanvasObject={(n, ctx, scale) => {
        const r = 3 + Math.sqrt(n.degree ?? 0);
        const x = n.x ?? 0;
        const y = n.y ?? 0;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = colors.types[n.type] ?? colors.muted;
        ctx.fill();
        // matched/selected vertices get an ink ring (identity never relies on colour alone)
        if (n.matched || n.id === selected) {
          ctx.lineWidth = (n.id === selected ? 2.5 : 1.5) / scale;
          ctx.strokeStyle = colors.ink;
          ctx.stroke();
        }
        if (showAllLabels || scale > 1.6 || n.matched || n.id === selected) {
          ctx.font = `${11 / scale}px sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillStyle = colors.ink;
          ctx.fillText(n.name, x, y + r + 1.5 / scale);
        }
      }}
      nodePointerAreaPaint={(n, color, ctx) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(n.x ?? 0, n.y ?? 0, 5 + Math.sqrt(n.degree ?? 0), 0, 2 * Math.PI);
        ctx.fill();
      }}
    />
  );
}
