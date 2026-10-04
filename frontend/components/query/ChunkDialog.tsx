"use client";

import { useQuery } from "@tanstack/react-query";

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ApiError, api } from "@/lib/api";

/** Opens the source chunk for a citation (GET /api/chunks/{id}). */
export function ChunkDialog({
  chunkId,
  filename,
  onClose,
}: {
  chunkId: string | null;
  filename?: string;
  onClose: () => void;
}) {
  const chunk = useQuery({
    queryKey: ["chunk", chunkId],
    queryFn: () => api.chunk(chunkId!),
    enabled: !!chunkId,
  });
  return (
    <Dialog open={!!chunkId} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="font-mono text-sm">{chunkId}</DialogTitle>
          <DialogDescription>
            {chunk.data
              ? `${filename ?? chunk.data.document_id} · p. ${chunk.data.page}${chunk.data.page_end !== chunk.data.page ? `–${chunk.data.page_end}` : ""} · ${chunk.data.token_count} tokens · cluster ${chunk.data.cluster_id}`
              : filename}
          </DialogDescription>
        </DialogHeader>
        {chunk.isLoading && <p className="text-sm text-muted-foreground">Loading…</p>}
        {chunk.isError && (
          <p className="text-sm text-destructive">
            {chunk.error instanceof ApiError && chunk.error.status === 404
              ? "This chunk no longer exists: its document was deleted or re-chunked since this run."
              : chunk.error.message}
          </p>
        )}
        {chunk.data && (
          <div className="max-h-[60vh] overflow-y-auto rounded bg-muted p-3 text-sm whitespace-pre-wrap">
            {chunk.data.text}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
