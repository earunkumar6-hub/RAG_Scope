"use client";

import { useQuery } from "@tanstack/react-query";
import { History } from "lucide-react";
import { useState } from "react";
import { cn } from "cn";

import { StatusIcon } from "@/components/pipeline/StageCard";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { api } from "@/lib/api";
import type { RunKind, StageStatus } from "@/lib/stages";

const RUN_STATUS: Record<string, StageStatus> = {
  running: "running",
  success: "success",
  warning: "warning",
  error: "error",
  blocked: "blocked",
};

/** Past runs of ``kind``; picking one re-opens its full stage trace. */
export function RunHistory({
  kind,
  currentId,
  onSelect,
}: {
  kind: RunKind;
  currentId: string | null;
  onSelect: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const runs = useQuery({ queryKey: ["runs", kind], queryFn: () => api.runs(kind), enabled: open });
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger render={<Button variant="outline" size="sm" />}>
        <History /> History
      </SheetTrigger>
      <SheetContent side="right" className="w-80">
        <SheetHeader>
          <SheetTitle>{kind === "ingest" ? "Ingestion runs" : "Query runs"}</SheetTitle>
        </SheetHeader>
        <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-4">
          {runs.isPending && <p className="px-2 text-xs text-muted-foreground">Loading…</p>}
          {runs.isError && <p className="px-2 text-xs text-destructive">{runs.error.message}</p>}
          {runs.data?.items.length === 0 && <p className="px-2 text-xs text-muted-foreground">No runs yet.</p>}
          {runs.data?.items.map((r) => (
            <button
              key={r.id}
              type="button"
              onClick={() => {
                onSelect(r.id);
                setOpen(false);
              }}
              className={cn(
                "flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-muted",
                r.id === currentId && "bg-muted",
              )}
            >
              <StatusIcon status={RUN_STATUS[r.status] ?? "pending"} className="mt-0.5" />
              <span className="min-w-0 flex-1">
                <span className="line-clamp-2 text-xs">{r.label || r.id}</span>
                <span className="text-[10px] text-muted-foreground">{new Date(r.created_at + (r.created_at.endsWith("Z") || r.created_at.includes("+") ? "" : "Z")).toLocaleString()}</span>
              </span>
            </button>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  );
}
