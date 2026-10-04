import { describe, expect, it } from "vitest";

import { effectiveMax, INGEST_PARAMS, QUERY_PARAMS, validateIngest, validateQuery } from "@/lib/params";
import { IngestParamsSchema, QueryParamsSchema } from "@/lib/schemas.generated";

describe("slider ranges match the generated (backend) schemas", () => {
  for (const [metas, schema] of [
    [INGEST_PARAMS, IngestParamsSchema],
    [QUERY_PARAMS, QueryParamsSchema],
  ] as const) {
    for (const m of metas) {
      it(m.key, () => {
        if (m.kind === "switch") {
          expect(schema.safeParse({ [m.key]: false }).success).toBe(true);
          return;
        }
        expect(schema.safeParse({ [m.key]: m.min }).success).toBe(true);
        expect(schema.safeParse({ [m.key]: m.min - m.step }).success).toBe(false);
        if (m.kind === "number") return; // seed has no upper bound on the backend
        if (m.key !== "chunk_overlap") {
          expect(schema.safeParse({ [m.key]: m.max }).success).toBe(true);
          expect(schema.safeParse({ [m.key]: m.max + m.step }).success).toBe(false);
        }
      });
    }
  }
});

describe("cross-field rules", () => {
  it("top_n must be <= top_k", () => {
    expect(validateQuery({ top_k: 3, top_n: 4 }).top_n).toMatch(/Top K \(3\)/);
    expect(validateQuery({ top_k: 4, top_n: 4 })).toEqual({});
  });

  it("chunk_overlap must be <= chunk_size / 2", () => {
    expect(validateIngest({ chunk_size: 256, chunk_overlap: 129 }).chunk_overlap).toMatch(/128/);
    expect(validateIngest({ chunk_size: 256, chunk_overlap: 128 })).toEqual({});
    expect(effectiveMax(INGEST_PARAMS[1], { chunk_size: 300 })).toBe(150);
  });

  it("reports range errors from the schema", () => {
    expect(validateQuery({ top_k: 51 }).top_k).toBeTruthy();
    expect(validateIngest({ chunk_size: 100 }).chunk_size).toBeTruthy();
  });
});
