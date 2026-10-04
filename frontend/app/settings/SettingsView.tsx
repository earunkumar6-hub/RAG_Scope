"use client";

// Server-wide settings (PUT /api/config): default pipeline parameters and guardrails. Model
// providers are read-only here: keys and providers come from environment variables only.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RotateCcw, Save, Undo2 } from "lucide-react";
import { useState } from "react";
import { cn } from "cn";

import { Header } from "@/components/layout/Header";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ApiError, api, type FieldError } from "@/lib/api";
import { INGEST_PARAMS, QUERY_PARAMS, validateIngest, validateQuery, type ParamValues } from "@/lib/params";
import { RuntimeConfigSchema, type RuntimeConfig } from "@/lib/schemas.generated";

/* eslint-disable @typescript-eslint/no-explicit-any -- config is edited generically by path */

const GUARD_HELP: Record<string, string> = {
  length_check: "Block queries longer than the token limit",
  prompt_injection: "Block injection attempts: regex patterns, plus LLM classifier score at or above the threshold",
  pii_detection: "Mask email, phone, card, Aadhaar and PAN before retrieval and logging",
  toxicity: "Block when the LLM classifier's toxicity score is at or above the threshold",
  off_topic: "Warn when the query's best similarity to any topic cluster is below the threshold",
  groundedness: "LLM judge: regenerate once, then fall back, if too many sentences are unsupported",
  citation_check: "Cited chunk ids must be in the context: regenerate once, then strip",
  pii_leak: "Redact PII in the answer",
  no_answer_handling: "Replace 'not in the documents' answers with the standard message",
  format_check: "Repair unbalanced markdown and cut answers over the length limit",
};
const FIELD: Record<string, { label: string; min: number; max: number; step: number }> = {
  threshold: { label: "Threshold", min: 0, max: 1, step: 0.05 },
  max_tokens: { label: "Max tokens", min: 8, max: 4096, step: 8 },
  max_chars: { label: "Max characters", min: 100, max: 50000, step: 100 },
  max_unsupported_ratio: { label: "Max unsupported ratio", min: 0, max: 1, step: 0.05 },
};
const PARAM_METAS = [...INGEST_PARAMS, ...QUERY_PARAMS].filter((m) => m.kind !== "switch");

function setPath(obj: any, path: string[], value: unknown): any {
  const [head, ...rest] = path;
  return { ...obj, [head]: rest.length ? setPath(obj[head], rest, value) : value };
}

function errorsByPath(draft: RuntimeConfig): Record<string, string> {
  const out: Record<string, string> = {};
  const parsed = RuntimeConfigSchema.safeParse(draft);
  for (const issue of parsed.success ? [] : parsed.error.issues) out[issue.path.join(".")] ??= issue.message;
  const d = draft.defaults as ParamValues;
  for (const [k, msg] of Object.entries({ ...validateIngest(d), ...validateQuery(d) }))
    out[`defaults.${k}`] ??= msg as string;
  return out;
}

function NumberField({
  id,
  label,
  value,
  min,
  max,
  step,
  error,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max?: number;
  step: number;
  error?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <Label htmlFor={id} className="text-xs">
        {label}
        <span className="ml-1 font-normal text-muted-foreground">
          ({min}–{max ?? "∞"})
        </span>
      </Label>
      <Input
        id={id}
        type="number"
        className="h-8 w-32 text-xs"
        value={Number.isFinite(value) ? value : ""}
        min={min}
        max={max}
        step={step}
        aria-invalid={!!error}
        onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
      />
      {error && <p className="text-[11px] text-destructive">{error}</p>}
    </div>
  );
}

function Section({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3 rounded-lg border bg-card p-4">
      <div>
        <h2 className="text-sm font-semibold">{title}</h2>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      {children}
    </section>
  );
}

function Providers() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const rows = ["llm", "embedding_model", "graph_store", "chroma"] as const;
  return (
    <Section
      title="Model providers"
      description="Read-only. Providers, models and API keys come from environment variables (.env): LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_PROVIDER, RERANKER_MODEL, NEO4J_URI. Restart the backend after changing them."
    >
      {health.isError && <p className="text-xs text-destructive">{health.error.message}</p>}
      <dl className="grid gap-x-6 gap-y-2 text-xs sm:grid-cols-2">
        {rows.map((name) => {
          const c = health.data?.components[name];
          return (
            <div key={name}>
              <dt className="font-medium">{name.replace(/_/g, " ")}</dt>
              <dd className="text-muted-foreground">
                {c ? `${c.status}: ${c.detail}` : "…"}
                {c &&
                  Object.entries(c.info)
                    .filter(([k]) => ["provider", "model", "backend", "uri", "collection"].includes(k))
                    .map(([k, v]) => (
                      <span key={k} className="block font-mono">
                        {k}: {v}
                      </span>
                    ))}
              </dd>
            </div>
          );
        })}
      </dl>
    </Section>
  );
}

export function SettingsView() {
  const queryClient = useQueryClient();
  const config = useQuery({ queryKey: ["config"], queryFn: api.config });
  const [draft, setDraft] = useState<RuntimeConfig | null>(null);
  const [serverErrors, setServerErrors] = useState<FieldError[]>([]);
  const [saved, setSaved] = useState(false);
  const current = draft ?? config.data ?? null;

  const save = useMutation({
    mutationFn: (c: RuntimeConfig) => api.saveConfig(c),
    onSuccess: (c) => {
      queryClient.setQueryData(["config"], c);
      setDraft(null);
      setServerErrors([]);
      setSaved(true);
    },
    onError: (e) => setServerErrors(e instanceof ApiError ? e.details : [{ field: "", message: String(e) }]),
  });

  if (!current)
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <Header title="Settings" />
        <p className="p-4 text-sm text-muted-foreground">{config.isError ? config.error.message : "Loading…"}</p>
      </div>
    );

  const errors = errorsByPath(current);
  for (const e of serverErrors) errors[e.field] ??= e.message;
  const dirty = draft !== null;
  const update = (path: string[], value: unknown) => {
    setDraft(setPath(current, path, value));
    setSaved(false);
  };
  const defaults = RuntimeConfigSchema.parse({}) as RuntimeConfig;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <Header title="Settings" />
      <main className="flex-1 space-y-4 overflow-y-auto p-4">
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => save.mutate(current)} disabled={!dirty || Object.keys(errors).length > 0 || save.isPending}>
            <Save /> {save.isPending ? "Saving…" : "Save"}
          </Button>
          <Button variant="outline" onClick={() => setDraft(null)} disabled={!dirty}>
            <Undo2 /> Discard changes
          </Button>
          <Dialog>
            <DialogTrigger render={<Button variant="outline" />}>
              <RotateCcw /> Reset to defaults
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Reset all settings?</DialogTitle>
                <DialogDescription>
                  Default parameters, every guardrail and online evaluation return to their built-in values and are saved
                  immediately. Documents, vectors and the knowledge graph are not affected.
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <DialogClose render={<Button variant="outline" />}>Cancel</DialogClose>
                <DialogClose render={<Button variant="destructive" />} onClick={() => save.mutate(defaults)}>
                  Reset
                </DialogClose>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          <span className={cn("text-xs", Object.keys(errors).length ? "text-destructive" : "text-muted-foreground")}>
            {Object.keys(errors).length
              ? "Fix the highlighted fields to save."
              : dirty
                ? "Unsaved changes."
                : saved
                  ? "Saved. New runs use these settings."
                  : "Applies server-wide to new runs. Per-browser overrides in the Parameters panel still win."}
          </span>
        </div>
        {errors[""] && <p className="text-xs text-destructive">{errors[""]}</p>}

        <Section
          title="Default parameters"
          description="Server-wide defaults for every run. The Parameters panel on Ingest/Query overrides them per browser."
        >
          <div className="flex flex-wrap gap-4">
            {PARAM_METAS.map((m) => (
              <NumberField
                key={m.key}
                id={`default-${m.key}`}
                label={m.label}
                value={(current.defaults as any)[m.key]}
                min={m.min}
                max={m.kind === "number" ? undefined : m.max}
                step={m.step}
                error={errors[`defaults.${m.key}`]}
                onChange={(v) => update(["defaults", m.key], v)}
              />
            ))}
          </div>
        </Section>

        {(["input", "output"] as const).map((side) => (
          <Section
            key={side}
            title={side === "input" ? "Input guardrails (Q2)" : "Output guardrails (Q9)"}
            description={
              side === "input"
                ? "Checked before retrieval. Blocking checks stop the run with the reason shown."
                : "Checked after generation. The streamed answer may be corrected; the change is shown as a diff."
            }
          >
            <div className="divide-y">
              {Object.entries((current.guardrails as any)[side] as Record<string, any>).map(([name, guard]) => (
                <div key={name} className="flex flex-wrap items-start gap-4 py-3">
                  <div className="flex w-64 items-start gap-2">
                    <Switch
                      id={`${side}-${name}`}
                      checked={guard.enabled}
                      onCheckedChange={(checked) => update(["guardrails", side, name, "enabled"], checked)}
                    />
                    <div>
                      <Label htmlFor={`${side}-${name}`} className="text-xs font-medium">
                        {name.replace(/_/g, " ")}
                      </Label>
                      <p className="text-[11px] text-muted-foreground">{GUARD_HELP[name]}</p>
                    </div>
                  </div>
                  {Object.entries(guard)
                    .filter(([k]) => k !== "enabled" && FIELD[k])
                    .map(([k, v]) => (
                      <NumberField
                        key={k}
                        id={`${side}-${name}-${k}`}
                        label={FIELD[k].label}
                        value={v as number}
                        min={FIELD[k].min}
                        max={FIELD[k].max}
                        step={FIELD[k].step}
                        error={errors[`guardrails.${side}.${name}.${k}`]}
                        onChange={(nv) => update(["guardrails", side, name, k], nv)}
                      />
                    ))}
                </div>
              ))}
            </div>
          </Section>
        ))}

        <Section
          title="Online evaluation (Q10)"
          description="Scores every query after its answer is shown. Faithfulness reuses Q9's judgement when it can; context precision adds one LLM judge call per query (cached per query and context)."
        >
          <div className="flex items-start gap-2 py-1">
            <Switch
              id="evaluation-online"
              checked={current.evaluation.online.enabled}
              onCheckedChange={(checked) => update(["evaluation", "online", "enabled"], checked)}
            />
            <Label htmlFor="evaluation-online" className="text-xs font-medium">
              online evaluation
            </Label>
          </div>
        </Section>

        <Providers />
      </main>
    </div>
  );
}
