"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Crosshair, Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import { cn } from "cn";

import { GraphView, TYPE_SLOTS, typeColor, type GraphVertex } from "@/components/graph/GraphView";
import { Header } from "@/components/layout/Header";
import { ChunkDialog } from "@/components/query/ChunkDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { api } from "@/lib/api";

const LIMITS = [100, 200, 500];

function VertexPanel({
  vertex,
  onFocus,
  onClose,
}: {
  vertex: GraphVertex;
  onFocus: () => void;
  onClose: () => void;
}) {
  const [chunk, setChunk] = useState<string | null>(null);
  return (
    <aside className="w-72 shrink-0 space-y-3 overflow-y-auto border-l p-3 text-sm">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold break-words">{vertex.name}</h3>
          <Badge variant="outline" className="mt-1">{vertex.type.toLowerCase()}</Badge>
        </div>
        <Button variant="ghost" size="icon-sm" aria-label="Close details" onClick={onClose}>
          <X />
        </Button>
      </div>
      {vertex.description && <p className="text-xs">{vertex.description}</p>}
      {vertex.aliases && vertex.aliases.length > 0 && (
        <p className="text-xs text-muted-foreground">Also: {vertex.aliases.join(", ")}</p>
      )}
      <p className="text-xs text-muted-foreground">{vertex.degree ?? 0} relation(s)</p>
      <Button variant="outline" size="sm" onClick={onFocus}>
        <Crosshair /> Focus neighbourhood
      </Button>
      <div className="space-y-1">
        <h4 className="text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
          Source chunks ({vertex.chunk_ids?.length ?? 0})
        </h4>
        <ul className="space-y-0.5">
          {(vertex.chunk_ids ?? []).map((id) => (
            <li key={id}>
              <button
                type="button"
                onClick={() => setChunk(id)}
                className="font-mono text-xs underline decoration-muted-foreground/50 underline-offset-2 hover:decoration-foreground"
              >
                {id}
              </button>
            </li>
          ))}
        </ul>
      </div>
      <ChunkDialog chunkId={chunk} onClose={() => setChunk(null)} />
    </aside>
  );
}

export function GraphExplorer() {
  const [draft, setDraft] = useState("");
  const [entity, setEntity] = useState("");
  const [hops, setHops] = useState(1);
  const [types, setTypes] = useState<string[]>([]);
  const [limit, setLimit] = useState(200);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const graph = useQuery({
    queryKey: ["graph", entity, hops, types, limit],
    queryFn: () => api.graph({ entity: entity || undefined, hops, types, limit }),
    placeholderData: keepPreviousData,
  });
  const data = graph.data;
  const selected = data?.vertices.find((v) => v.id === selectedId) ?? null;
  // Stable identity per response: a new array would restart the force layout on every render.
  const vertices = useMemo(
    () => (data ? data.vertices.map((v) => ({ ...v, matched: data.seeds.includes(v.id) })) : []),
    [data],
  );
  const allTypes = data ? Object.keys(data.stats.types) : [];

  const search = (value: string) => {
    setEntity(value.trim());
    setSelectedId(null);
  };
  const toggleType = (t: string) =>
    setTypes((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <Header title="Knowledge graph" />
      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col gap-3 overflow-y-auto p-4">
          <form
            className="flex flex-wrap items-end gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              search(draft);
            }}
          >
            <div className="flex min-w-56 flex-1 gap-2">
              <Input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="Search an entity (name or alias)…"
                aria-label="Entity search"
              />
              <Button type="submit" variant="outline">
                <Search /> Search
              </Button>
              {entity && (
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => {
                    setDraft("");
                    search("");
                  }}
                >
                  Clear
                </Button>
              )}
            </div>
            <div className="w-40 space-y-1">
              <Label className="text-xs">
                Hops: {hops} {!entity && <span className="text-muted-foreground">(with a search)</span>}
              </Label>
              <Slider
                aria-label="Hops"
                value={[hops]}
                min={0}
                max={3}
                step={1}
                disabled={!entity}
                onValueChange={(v) => setHops(Array.isArray(v) ? v[0] : v)}
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Show up to</Label>
              <div className="flex gap-1">
                {LIMITS.map((n) => (
                  <Button
                    key={n}
                    type="button"
                    size="xs"
                    variant={limit === n ? "secondary" : "ghost"}
                    onClick={() => setLimit(n)}
                  >
                    {n}
                  </Button>
                ))}
              </div>
            </div>
          </form>

          {allTypes.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              <span className="text-muted-foreground">Entity types:</span>
              {TYPE_SLOTS.filter((t) => allTypes.includes(t)).map((t) => (
                <button
                  key={t}
                  type="button"
                  aria-pressed={types.includes(t)}
                  onClick={() => toggleType(t)}
                  className={cn(
                    "flex items-center gap-1 rounded-full border px-2 py-0.5 hover:border-ring",
                    types.includes(t) && "border-primary bg-primary/10",
                  )}
                >
                  <span className="size-2 rounded-full" style={{ background: typeColor(t) }} />
                  {t.toLowerCase()} {data?.stats.types[t]}
                </button>
              ))}
              {types.length > 0 && (
                <Button size="xs" variant="ghost" onClick={() => setTypes([])}>
                  all types
                </Button>
              )}
            </div>
          )}

          {graph.isError && <p className="text-sm text-destructive">{graph.error.message}</p>}
          {data && data.stats.vertices === 0 ? (
            <p className="text-sm text-muted-foreground">
              The knowledge graph is empty. Ingest documents with &quot;Build knowledge graph&quot; on
              (S8 needs a configured LLM).
            </p>
          ) : (
            data && (
              <>
                <p className="text-xs text-muted-foreground">
                  Showing {data.vertices.length} of {data.stats.vertices} vertices and{" "}
                  {data.edges.length} of {data.stats.edges} edges
                  {entity &&
                    (data.seeds.length
                      ? ` around ${data.seeds.length} match(es) for "${entity}"`
                      : ` - no entity matches "${entity}"`)}
                  . Click a vertex for details.
                </p>
                <GraphView
                  vertices={vertices}
                  edges={data.edges}
                  height={560}
                  selected={selectedId}
                  onSelect={(v) => setSelectedId(v.id)}
                />
              </>
            )
          )}
        </main>
        {selected && (
          <VertexPanel
            vertex={selected}
            onClose={() => setSelectedId(null)}
            onFocus={() => {
              setDraft(selected.name);
              setEntity(selected.name);
              setHops((h) => Math.max(h, 1));
            }}
          />
        )}
      </div>
    </div>
  );
}
