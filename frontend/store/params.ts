"use client";

// Parameter overrides chosen in the Parameters Panel. Only keys the user changed are stored;
// everything else falls back to the server defaults (GET /api/config). Persisted per browser.
import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { ParamKey, ParamValues } from "@/lib/params";

type ParamsStore = {
  overrides: ParamValues;
  set: (key: ParamKey, value: number | boolean) => void;
  reset: (keys: ParamKey[]) => void;
};

export const useParamsStore = create<ParamsStore>()(
  persist(
    (set) => ({
      overrides: {},
      set: (key, value) => set((s) => ({ overrides: { ...s.overrides, [key]: value } })),
      reset: (keys) =>
        set((s) => ({
          overrides: Object.fromEntries(
            Object.entries(s.overrides).filter(([k]) => !keys.includes(k as ParamKey)),
          ),
        })),
    }),
    { name: "graphrag-params" },
  ),
);

export function effectiveParams(defaults: ParamValues | undefined, overrides: ParamValues): ParamValues {
  return { ...(defaults ?? {}), ...overrides };
}
