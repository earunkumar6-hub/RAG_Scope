"use client";

import { StatusIcon, displayStatus, statusLabel } from "@/components/pipeline/StageCard";
import { KeyValues } from "@/components/pipeline/KeyValues";
import { STAGE_VIEWS } from "@/components/pipeline/stage-views";
import type { StageEvent } from "@/lib/schemas.generated";
import { formatMs, isSkipped, type RunState, type StageDef } from "@/lib/stages";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-1.5">
      <h4 className="text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">{title}</h4>
      {children}
    </section>
  );
}

/** Full detail for one stage: inputs, params used, duration, outputs (stage view + raw JSON). */
export function StageDetail({ def, run }: { def: StageDef; run: RunState }) {
  const ev: StageEvent | undefined = run.stages[def.id];
  const status = displayStatus(def, ev, !!run.done);
  const View = STAGE_VIEWS[def.id];
  const hasOutput = ev && !isSkipped(ev) && ev.status !== "pending" && ev.status !== "running";
  return (
    <div className="space-y-4 rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-center gap-2">
        <StatusIcon status={status} />
        <h3 className="text-sm font-semibold">{def.name}</h3>
        <span className="text-xs text-muted-foreground">{statusLabel(status)}</span>
        {ev?.duration_ms != null && (
          <span className="ml-auto text-xs tabular-nums text-muted-foreground">{formatMs(ev.duration_ms)}</span>
        )}
      </div>
      {(def.notBuilt || ev?.summary) && <p className="text-sm">{def.notBuilt ?? ev?.summary}</p>}

      <div className="grid gap-4 md:grid-cols-2">
        <Section title="Inputs">
          <p className="text-xs">{def.inputs}</p>
        </Section>
        <Section title="Params used">
          <KeyValues values={ev?.params_used ?? {}} />
        </Section>
      </div>

      {hasOutput && (
        <Section title="Outputs">
          {View ? <View data={ev.data} run={run} /> : null}
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground">Raw stage data</summary>
            <pre className="mt-2 max-h-80 overflow-auto rounded bg-muted p-2 font-mono text-[11px]">
              {JSON.stringify(ev.data, null, 2)}
            </pre>
          </details>
        </Section>
      )}
    </div>
  );
}
