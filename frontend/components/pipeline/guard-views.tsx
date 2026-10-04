"use client";

// Q2 / Q9 guardrail views: one chip per check (icon + label + reason, never colour alone).
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  EyeOff,
  MinusCircle,
  RefreshCw,
  Replace,
  Wrench,
} from "lucide-react";
import { useState } from "react";
import { cn } from "cn";

import { Button } from "@/components/ui/button";
import { diffWords } from "@/lib/diff";
import type { RunState } from "@/lib/stages";

/* eslint-disable @typescript-eslint/no-explicit-any -- stage payloads are free-form JSON */
type Data = Record<string, any>;
type Props = { data: Data; run: RunState };

const VERDICT: Record<string, { icon: typeof Ban; tone: string; label: string }> = {
  pass: { icon: CheckCircle2, tone: "text-emerald-600 dark:text-emerald-500", label: "pass" },
  warn: { icon: AlertTriangle, tone: "text-amber-600 dark:text-amber-500", label: "warn" },
  mask: { icon: EyeOff, tone: "text-sky-600 dark:text-sky-400", label: "masked" },
  block: { icon: Ban, tone: "text-red-600 dark:text-red-500", label: "blocked" },
  skipped: { icon: MinusCircle, tone: "text-muted-foreground", label: "skipped" },
  regenerated: { icon: RefreshCw, tone: "text-amber-600 dark:text-amber-500", label: "regenerated" },
  repaired: { icon: Wrench, tone: "text-amber-600 dark:text-amber-500", label: "repaired" },
  replaced: { icon: Replace, tone: "text-amber-600 dark:text-amber-500", label: "replaced" },
  redacted: { icon: EyeOff, tone: "text-amber-600 dark:text-amber-500", label: "redacted" },
};

export function GuardChecks({ checks }: { checks: Data[] }) {
  return (
    <ul className="space-y-1.5">
      {checks.map((c) => {
        const v = VERDICT[c.verdict] ?? VERDICT.skipped;
        const Icon = v.icon;
        return (
          <li key={c.name} className="flex items-start gap-2 rounded-md border px-2 py-1.5 text-xs">
            <Icon aria-hidden className={cn("mt-0.5 size-3.5 shrink-0", v.tone)} />
            <span className="w-36 shrink-0 font-medium">{c.name.replace(/_/g, " ")}</span>
            <span className={cn("w-20 shrink-0 font-medium", v.tone)}>{v.label}</span>
            <span className="min-w-0 flex-1 text-muted-foreground">
              {c.reason}
              {c.score != null && c.threshold != null && (
                <span className="ml-1 tabular-nums">
                  (score {Number(c.score).toFixed(2)} / threshold {Number(c.threshold).toFixed(2)})
                </span>
              )}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export function AnswerDiff({ before, after }: { before: string; after: string }) {
  // The stored original is PII-redacted, so a PII-only correction leaves nothing to diff.
  if (before === after)
    return (
      <p className="rounded bg-muted p-2 text-xs text-muted-foreground">
        Only personal data was redacted. The original answer is not stored, so there is nothing else to compare.
      </p>
    );
  return (
    <p className="rounded bg-muted p-2 text-xs leading-relaxed whitespace-pre-wrap">
      {diffWords(before, after).map((p, i) =>
        p.type === "same" ? (
          <span key={i}>{p.text}</span>
        ) : p.type === "removed" ? (
          <del key={i} className="bg-red-500/15 text-red-700 decoration-red-500/60 dark:text-red-400">
            {p.text}
          </del>
        ) : (
          <ins key={i} className="bg-emerald-500/15 text-emerald-700 no-underline dark:text-emerald-400">
            {p.text}
          </ins>
        ),
      )}
    </p>
  );
}

export function Q2View({ data }: Props) {
  return (
    <div className="space-y-2 text-xs">
      {data.query_used && (
        <p className="text-muted-foreground">
          Query as checked (PII already masked): <span className="text-foreground">{data.query_used}</span>
        </p>
      )}
      <GuardChecks checks={data.checks ?? []} />
    </div>
  );
}

export function Q9View({ data }: Props) {
  const [showDiff, setShowDiff] = useState(true);
  return (
    <div className="space-y-2 text-xs">
      <GuardChecks checks={data.checks ?? []} />
      {data.modified && (
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">
              Answer changed{data.regenerated ? " (regenerated once)" : ""}
            </span>
            <Button size="xs" variant="ghost" onClick={() => setShowDiff((v) => !v)}>
              {showDiff ? "Hide" : "Show"} original vs final
            </Button>
          </div>
          {showDiff && <AnswerDiff before={data.original_answer} after={data.final_answer} />}
        </div>
      )}
      {data.judge?.length > 0 && (
        <details>
          <summary className="cursor-pointer text-muted-foreground">
            Groundedness judge ({data.judge.length} sentence(s), last pass)
          </summary>
          <ul className="mt-1 space-y-0.5">
            {data.judge.map((s: Data, i: number) => (
              <li key={i} className="flex gap-2">
                <span className={s.supported ? "text-emerald-600 dark:text-emerald-500" : "text-red-600 dark:text-red-500"}>
                  {s.supported ? "supported" : "unsupported"}
                </span>
                <span className="text-muted-foreground">{s.text}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
