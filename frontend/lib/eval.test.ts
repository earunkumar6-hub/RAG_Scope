import { describe, expect, it } from "vitest";

import { cellLabel, parseGrid, varyingKeys } from "@/lib/eval";

describe("parseGrid", () => {
  it("skips empty inputs and multiplies value counts into cells", () => {
    const out = parseGrid({ top_k: "5, 10", top_n: "2,3,4", seed: " " }, 6);
    expect(out.grid).toEqual({ top_k: [5, 10], top_n: [2, 3, 4] });
    expect(out.cells).toBe(6);
    expect(out.errors).toEqual({});
  });

  it("reports bad values per parameter", () => {
    const out = parseGrid({ top_k: "5,x", top_n: "2,2", temperature: "0,0.1,0.2" }, 2);
    expect(out.errors).toEqual({
      top_k: "numbers separated by commas",
      top_n: "values must be distinct",
      temperature: "at most 2 values",
    });
    expect(out.cells).toBe(1);
  });
});

describe("cell labels", () => {
  it("names only the parameters that vary", () => {
    const cells = [
      { top_k: 5, top_n: 2 },
      { top_k: 10, top_n: 2 },
    ];
    const keys = varyingKeys(cells);
    expect(keys).toEqual(["top_k"]);
    expect(cellLabel(cells[1], keys)).toBe("top_k=10");
    expect(cellLabel(cells[0], varyingKeys([cells[0]]))).toBe("defaults");
  });
});
