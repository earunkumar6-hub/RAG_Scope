"use client";

import { AlertTriangle, Ban, CheckCircle2, Circle, CircleDashed, Loader2, MinusCircle, XCircle } from "lucide-react";
import { cn } from "cn";

import type { StageEvent } from "@/lib/schemas.generated";
import { formatMs, isSkipped, type StageDef } from "@/lib/stages";

export type DisplayStatus = StageEvent["status"] | "skipped" | "not_built" | "not_run" | "idle";

/** ``finished``: the run is over, so stages still pending were never reached (an earlier stage
 * failed) rather than queued. */
export function displayStatus(def: StageDef, ev: StageEvent | undefined, finished = false): DisplayStatus {
  if (def.notBuilt) return "not_built";
  if (finished && (!ev || ev.status === "pending" || ev.status === "running")) return "not_run";
  if (!ev) return "idle";
  if (ev.status === "success" && isSkipped(ev)) return "skipped";
  return ev.status;
}

const ICONS: Record<DisplayStatus, { icon: typeof Circle; className: string; label: string }> = {
  idle: { icon: Circle, className: "text-muted-foreground/50", label: "Not started" },
  pending: { icon: Circle, className: "text-muted-foreground", label: "Queued" },
  running: { icon: Loader2, className: "animate-spin text-primary", label: "Running" },
  success: { icon: CheckCircle2, className: "text-emerald-600 dark:text-emerald-500", label: "Success" },
  warning: { icon: AlertTriangle, className: "text-amber-600 dark:text-amber-500", label: "Warning" },
  blocked: { icon: Ban, className: "text-red-600 dark:text-red-500", label: "Blocked" },
  error: { icon: XCircle, className: "text-red-600 dark:text-red-500", label: "Error" },
  skipped: { icon: MinusCircle, className: "text-muted-foreground", label: "Skipped" },
  not_run: { icon: MinusCircle, className: "text-muted-foreground/60", label: "Not run" },
  not_built: { icon: CircleDashed, className: "text-muted-foreground/60", label: "Not built yet" },
};

export function StatusIcon({ status, className }: { status: DisplayStatus; className?: string }) {
  const { icon: Icon, className: tone, label } = ICONS[status];
  return <Icon aria-label={label} className={cn("size-4 shrink-0", tone, className)} />;
}

export function statusLabel(status: DisplayStatus): string {
  return ICONS[status].label;
}

export function StageCard({
  def,
  ev,
  index,
  selected,
  finished,
  onSelect,
}: {
  def: StageDef;
  ev: StageEvent | undefined;
  index: number;
  selected: boolean;
  finished: boolean;
  onSelect: () => void;
}) {
  const status = displayStatus(def, ev, finished);
  const summary = def.notBuilt ?? (status !== "not_run" && ev && ev.status !== "pending" ? ev.summary : "");
  const muted = ["not_built", "not_run", "skipped", "idle"].includes(status);
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "flex w-full flex-col gap-1 rounded-lg border bg-card p-2.5 text-left transition-colors hover:border-ring",
        selected && "border-primary ring-1 ring-primary",
        muted && "opacity-70",
      )}
    >
      <div className="flex items-center gap-1.5">
        <StatusIcon status={status} />
        <span className="text-[10px] text-muted-foreground">{index + 1}</span>
        <span className="truncate text-xs font-medium">{def.name}</span>
        <span className="ml-auto shrink-0 text-[10px] tabular-nums text-muted-foreground">
          {formatMs(ev?.duration_ms)}
        </span>
      </div>
      <p className="line-clamp-2 min-h-[2lh] text-[11px] text-muted-foreground">{summary || statusLabel(status)}</p>
    </button>
  );
}
