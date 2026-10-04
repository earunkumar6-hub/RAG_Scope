"use client";

import { useQuery } from "@tanstack/react-query";
import { PanelRightClose, PanelRightOpen, RotateCcw } from "lucide-react";
import { useState } from "react";

import { ParamSlider } from "@/components/params/ParamSlider";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useMediaQuery } from "@/lib/useMediaQuery";
import { pickParams, type ParamErrors, type ParamMeta, type ParamValues } from "@/lib/params";
import { effectiveParams, useParamsStore } from "@/store/params";

/** Effective values (server defaults + local overrides) and validation errors for ``metas``. */
export function useRunParams(metas: ParamMeta[], validate: (v: ParamValues) => ParamErrors) {
  const config = useQuery({ queryKey: ["config"], queryFn: api.config });
  const overrides = useParamsStore((s) => s.overrides);
  const values = effectiveParams(config.data?.defaults as ParamValues | undefined, overrides);
  const errors = config.data ? validate(values) : {};
  return {
    config,
    values,
    errors,
    valid: !!config.data && Object.keys(errors).length === 0,
    params: pickParams(metas, values),
  };
}

export function ParametersPanel({
  title,
  metas,
  validate,
}: {
  title: string;
  metas: ParamMeta[];
  validate: (v: ParamValues) => ParamErrors;
}) {
  // Open by default only where there is room; an explicit toggle by the user wins.
  const wide = useMediaQuery("(min-width: 1280px)");
  const [userOpen, setOpen] = useState<boolean | null>(null);
  const open = userOpen ?? wide;
  const { config, values, errors } = useRunParams(metas, validate);
  const overrides = useParamsStore((s) => s.overrides);
  const set = useParamsStore((s) => s.set);
  const reset = useParamsStore((s) => s.reset);
  const changed = metas.filter((m) => overrides[m.key] != null).map((m) => m.key);

  if (!open)
    return (
      <aside className="flex shrink-0 flex-col items-center border-l p-2">
        <Button variant="ghost" size="icon-sm" aria-label="Show parameters" onClick={() => setOpen(true)}>
          <PanelRightOpen />
        </Button>
      </aside>
    );

  return (
    <aside className="flex w-72 shrink-0 flex-col border-l">
      <div className="flex items-center gap-1 border-b px-3 py-2">
        <span className="text-sm font-semibold">{title}</span>
        <Button
          className="ml-auto"
          variant="ghost"
          size="xs"
          disabled={!changed.length}
          onClick={() => reset(changed)}
          title="Reset to server defaults"
        >
          <RotateCcw /> Reset
        </Button>
        <Button variant="ghost" size="icon-sm" aria-label="Hide parameters" onClick={() => setOpen(false)}>
          <PanelRightClose />
        </Button>
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto p-3">
        {config.isError && <p className="text-xs text-destructive">Could not load defaults: {config.error.message}</p>}
        {config.isPending && <p className="text-xs text-muted-foreground">Loading defaults…</p>}
        {config.data &&
          metas.map((m) => (
            <ParamSlider
              key={m.key}
              meta={m}
              values={values}
              changed={overrides[m.key] != null}
              defaultValue={(config.data.defaults as ParamValues)[m.key]}
              error={errors[m.key]}
              onChange={(v) => set(m.key, v)}
            />
          ))}
        <p className="text-[11px] text-muted-foreground">
          Changes apply to the next run and appear in each stage&apos;s params used.
        </p>
      </div>
    </aside>
  );
}
