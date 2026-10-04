"use client";

// Documents table -> chunk browser (cluster / duplicate filters), plus confirmed delete.
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Copy, Trash2 } from "lucide-react";
import { useState } from "react";
import { cn } from "cn";

import { Header } from "@/components/layout/Header";
import { ChunkDialog } from "@/components/query/ChunkDialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { api, type DocumentDeleted, type DocumentRow } from "@/lib/api";

const PAGE = 50;
const SELECT = "h-8 rounded-md border bg-background px-2 text-xs";

function formatSize(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function DeleteDialog({
  doc,
  onClose,
  onDeleted,
}: {
  doc: DocumentRow | null;
  onClose: () => void;
  onDeleted: (r: DocumentDeleted) => void;
}) {
  const queryClient = useQueryClient();
  const del = useMutation({
    mutationFn: (id: string) => api.deleteDocument(id),
    onSuccess: (r) => {
      for (const key of ["documents", "chunks", "clusters", "graph", "chunk"]) queryClient.invalidateQueries({ queryKey: [key] });
      onDeleted(r);
      onClose();
    },
  });
  return (
    <Dialog
      open={!!doc}
      onOpenChange={(open) => {
        if (!open) {
          del.reset();
          onClose();
        }
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete {doc?.filename}?</DialogTitle>
          <DialogDescription>
            Removes its {doc?.chunk_count} chunk(s) from the vector store, the knowledge graph and the database. Near-duplicates in
            other documents take over its place and graph facts. This cannot be undone; you can ingest the file again later.
          </DialogDescription>
        </DialogHeader>
        {del.isError && <p className="text-xs text-destructive">{del.error.message}</p>}
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>Cancel</DialogClose>
          <Button variant="destructive" disabled={del.isPending} onClick={() => doc && del.mutate(doc.id)}>
            <Trash2 /> {del.isPending ? "Deleting…" : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ChunkBrowser({ docs }: { docs: DocumentRow[] }) {
  const [documentId, setDocumentId] = useState("");
  const [clusterId, setClusterId] = useState("");
  const [dupes, setDupes] = useState("");
  const [offset, setOffset] = useState(0);
  const [openChunk, setOpenChunk] = useState<string | null>(null);
  const clusters = useQuery({ queryKey: ["clusters"], queryFn: api.clusters });
  const q = {
    document_id: documentId || undefined,
    cluster_id: clusterId === "" ? undefined : Number(clusterId),
    duplicates: dupes === "" ? undefined : dupes === "only",
    offset,
    limit: PAGE,
  };
  const chunks = useQuery({ queryKey: ["chunks", q], queryFn: () => api.chunks(q), placeholderData: keepPreviousData });
  const names = new Map(docs.map((d) => [d.id, d.filename]));
  const labels = new Map((clusters.data ?? []).map((c) => [c.id, c.label]));
  const total = chunks.data?.total ?? 0;
  const filter = (set: (v: string) => void) => (e: React.ChangeEvent<HTMLSelectElement>) => {
    set(e.target.value);
    setOffset(0);
  };

  return (
    <section className="space-y-3 rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-end gap-3">
        <h2 className="mr-auto text-sm font-semibold">Chunks</h2>
        <div className="space-y-1">
          <Label htmlFor="f-doc" className="text-xs">Document</Label>
          <select id="f-doc" className={SELECT} value={documentId} onChange={filter(setDocumentId)}>
            <option value="">All documents</option>
            {docs.map((d) => (
              <option key={d.id} value={d.id}>{d.filename}</option>
            ))}
          </select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="f-cluster" className="text-xs">Cluster</Label>
          <select id="f-cluster" className={SELECT} value={clusterId} onChange={filter(setClusterId)}>
            <option value="">All clusters</option>
            {(clusters.data ?? []).map((c) => (
              <option key={c.id} value={c.id}>{c.id}: {c.label} ({c.size})</option>
            ))}
          </select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="f-dup" className="text-xs">Duplicates</Label>
          <select id="f-dup" className={SELECT} value={dupes} onChange={filter(setDupes)}>
            <option value="">All chunks</option>
            <option value="only">Only near-duplicates</option>
            <option value="none">Only canonical</option>
          </select>
        </div>
      </div>
      {chunks.isError && <p className="text-xs text-destructive">{chunks.error.message}</p>}
      <div className="overflow-x-auto rounded border">
        <table className="w-full text-xs">
          <thead className="bg-muted text-left text-muted-foreground">
            <tr>
              {["Chunk", "Document", "Pages", "Tokens", "Cluster", "Text"].map((h) => (
                <th key={h} className="px-2 py-1 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className={cn("divide-y", chunks.isPlaceholderData && "opacity-60")}>
            {(chunks.data?.items ?? []).map((c) => (
              <tr key={c.id} className="cursor-pointer hover:bg-muted/50" onClick={() => setOpenChunk(c.id)}>
                <td className="px-2 py-1 align-top font-mono whitespace-nowrap">
                  {c.id}
                  {c.is_duplicate_of && (
                    <span className="mt-0.5 flex items-center gap-1 font-sans text-[11px] text-amber-700 dark:text-amber-500">
                      <Copy aria-hidden className="size-3" /> duplicate of{" "}
                      <button
                        type="button"
                        className="font-mono underline underline-offset-2"
                        onClick={(e) => {
                          e.stopPropagation();
                          setOpenChunk(c.is_duplicate_of);
                        }}
                      >
                        {c.is_duplicate_of}
                      </button>
                    </span>
                  )}
                </td>
                <td className="px-2 py-1 align-top">{names.get(c.document_id) ?? c.document_id}</td>
                <td className="px-2 py-1 align-top whitespace-nowrap">{c.page === c.page_end ? c.page : `${c.page}–${c.page_end}`}</td>
                <td className="px-2 py-1 align-top tabular-nums">{c.token_count}</td>
                <td className="px-2 py-1 align-top">{labels.get(c.cluster_id) ? `${c.cluster_id}: ${labels.get(c.cluster_id)}` : c.cluster_id}</td>
                <td className="max-w-md px-2 py-1 align-top text-muted-foreground">
                  <span className="line-clamp-2">{c.text}</span>
                </td>
              </tr>
            ))}
            {chunks.data && total === 0 && (
              <tr>
                <td colSpan={6} className="px-2 py-3 text-center text-muted-foreground">No chunks match these filters.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span>{total ? `${offset + 1}–${Math.min(offset + PAGE, total)} of ${total}` : ""}</span>
        <Button className="ml-auto" size="icon-sm" variant="outline" aria-label="Previous page" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
          <ChevronLeft />
        </Button>
        <Button size="icon-sm" variant="outline" aria-label="Next page" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>
          <ChevronRight />
        </Button>
      </div>
      <ChunkDialog
        chunkId={openChunk}
        filename={names.get(chunks.data?.items.find((c) => c.id === openChunk)?.document_id ?? "")}
        onClose={() => setOpenChunk(null)}
      />
    </section>
  );
}

export function DocumentsView() {
  const docs = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [toDelete, setToDelete] = useState<DocumentRow | null>(null);
  const [deleted, setDeleted] = useState<DocumentDeleted | null>(null);
  const rows = docs.data ?? [];

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <Header title="Documents" />
      <main className="flex-1 space-y-4 overflow-y-auto p-4">
        {deleted && (
          <p role="status" className="rounded-md border bg-muted/40 px-3 py-2 text-xs">
            Deleted <span className="font-medium">{deleted.filename}</span>: {deleted.chunks_deleted} chunk(s),{" "}
            {deleted.vertices_removed} graph vertices and {deleted.edges_removed} edges removed
            {deleted.duplicates_promoted > 0 && `; ${deleted.duplicates_promoted} near-duplicate(s) promoted and given its graph facts`}
            {deleted.clusters_removed > 0 && `; ${deleted.clusters_removed} empty cluster(s) dropped`}.
          </p>
        )}
        <section className="space-y-3 rounded-lg border bg-card p-4">
          <h2 className="text-sm font-semibold">Documents ({rows.length})</h2>
          {docs.isError && <p className="text-xs text-destructive">{docs.error.message}</p>}
          {docs.data && rows.length === 0 && (
            <p className="text-xs text-muted-foreground">No documents yet. Upload some on the Ingest page.</p>
          )}
          {rows.length > 0 && (
            <div className="overflow-x-auto rounded border">
              <table className="w-full text-xs">
                <thead className="bg-muted text-left text-muted-foreground">
                  <tr>
                    {["File", "Size", "Pages", "Tokens", "Chunks", "Duplicates", "Chunking", "Version", "Status", ""].map((h) => (
                      <th key={h} className="px-2 py-1 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {rows.map((d) => (
                    <tr key={d.id}>
                      <td className="px-2 py-1">
                        <span className="font-medium">{d.filename}</span>
                        <span className="block font-mono text-[11px] text-muted-foreground">{d.id.slice(0, 8)}</span>
                      </td>
                      <td className="px-2 py-1 whitespace-nowrap">{formatSize(d.size_bytes)}</td>
                      <td className="px-2 py-1 tabular-nums">{d.page_count}</td>
                      <td className="px-2 py-1 tabular-nums">{d.total_tokens.toLocaleString()}</td>
                      <td className="px-2 py-1 tabular-nums">{d.chunk_count}</td>
                      <td className="px-2 py-1 tabular-nums">{d.duplicate_chunk_count}</td>
                      <td className="px-2 py-1 whitespace-nowrap">{d.chunk_size} / {d.chunk_overlap}</td>
                      <td className="px-2 py-1 tabular-nums">v{d.version}</td>
                      <td className="px-2 py-1">
                        <Badge variant={d.status === "error" ? "destructive" : d.status === "ready" ? "secondary" : "outline"} title={d.error ?? undefined}>
                          {d.status}
                        </Badge>
                      </td>
                      <td className="px-2 py-1 text-right">
                        <Button size="xs" variant="ghost" aria-label={`Delete ${d.filename}`} onClick={() => setToDelete(d)}>
                          <Trash2 /> Delete
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        <ChunkBrowser docs={rows} />
      </main>
      <DeleteDialog doc={toDelete} onClose={() => setToDelete(null)} onDeleted={setDeleted} />
    </div>
  );
}
