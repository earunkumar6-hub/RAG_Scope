"use client";

import { StageCard } from "@/components/pipeline/StageCard";
import { StageDetail } from "@/components/pipeline/StageDetail";
import type { RunState, StageDef, StageId } from "@/lib/stages";

/** Horizontal: cards in a row with the selected stage's detail below.
 * Vertical: cards in a column, the selected one expanded in place. */
export function PipelineStepper({
  stages,
  run,
  selected,
  onSelect,
  orientation,
}: {
  stages: StageDef[];
  run: RunState;
  selected: StageId | null;
  onSelect: (id: StageId | null) => void;
  orientation: "horizontal" | "vertical";
}) {
  const toggle = (id: StageId) => onSelect(selected === id ? null : id);
  const selectedDef = stages.find((s) => s.id === selected);

  if (orientation === "vertical")
    return (
      <ol className="space-y-2">
        {stages.map((def, i) => (
          <li key={def.id} className="space-y-2">
            <StageCard def={def} ev={run.stages[def.id]} index={i} selected={selected === def.id} finished={!!run.done} onSelect={() => toggle(def.id)} />
            {selected === def.id && <StageDetail def={def} run={run} />}
          </li>
        ))}
      </ol>
    );

  return (
    <div className="@container space-y-3">
      {/* columns follow the stepper's own width (container query), not the viewport */}
      <ol className="grid grid-cols-2 gap-2 @lg:grid-cols-4 @6xl:grid-cols-8">
        {stages.map((def, i) => (
          <li key={def.id}>
            <StageCard def={def} ev={run.stages[def.id]} index={i} selected={selected === def.id} finished={!!run.done} onSelect={() => toggle(def.id)} />
          </li>
        ))}
      </ol>
      {selectedDef && <StageDetail def={selectedDef} run={run} />}
    </div>
  );
}
