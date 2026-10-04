"use client";

import { cn } from "cn";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { effectiveMax, type ParamMeta, type ParamValues } from "@/lib/params";

export function ParamSlider({
  meta,
  values,
  changed,
  defaultValue,
  error,
  onChange,
}: {
  meta: ParamMeta;
  values: ParamValues;
  changed: boolean;
  defaultValue: number | boolean | undefined;
  error?: string;
  onChange: (value: number | boolean) => void;
}) {
  const id = `param-${meta.key}`;
  if (meta.kind === "switch") {
    const on = values[meta.key] !== false; // unset = backend default (on)
    return (
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Label htmlFor={id} className="text-xs" title={meta.description}>
            {meta.label}
            {changed && <span className="ml-1 text-primary" title="changed from default">•</span>}
          </Label>
          <Switch id={id} className="ml-auto" checked={on} onCheckedChange={(checked) => onChange(checked)} />
        </div>
        <p className="text-[11px] text-muted-foreground">{meta.description}</p>
      </div>
    );
  }
  const value = (values[meta.key] as number | undefined) ?? meta.min;
  const max = effectiveMax(meta, values);
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-2">
        <Label htmlFor={id} className="text-xs" title={meta.description}>
          {meta.label}
          {changed && <span className="ml-1 text-primary" title={`default ${defaultValue}`}>•</span>}
        </Label>
        {meta.kind !== "number" && (
          <Input
            id={id}
            type="number"
            className="ml-auto h-6 w-20 px-1.5 text-right text-xs"
            value={value}
            min={meta.min}
            max={max}
            step={meta.step}
            aria-invalid={!!error}
            onChange={(e) => e.target.value !== "" && onChange(Number(e.target.value))}
          />
        )}
      </div>
      {meta.kind === "number" ? (
        <Input
          id={id}
          type="number"
          className="h-7 text-xs"
          value={value}
          min={meta.min}
          step={1}
          aria-invalid={!!error}
          onChange={(e) => e.target.value !== "" && onChange(Number(e.target.value))}
        />
      ) : (
        <>
          <Slider
            aria-label={meta.label}
            value={[Math.min(value, max)]}
            min={meta.min}
            max={max}
            step={meta.step}
            onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : v)}
          />
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>{meta.min}</span>
            <span>{max}</span>
          </div>
        </>
      )}
      <p className={cn("text-[11px]", error ? "text-destructive" : "text-muted-foreground")}>
        {error ?? meta.description}
      </p>
    </div>
  );
}
