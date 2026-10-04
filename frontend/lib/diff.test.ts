import { describe, expect, it } from "vitest";

import { diffWords } from "@/lib/diff";

describe("diffWords", () => {
  it("marks removed and added words, keeping the rest", () => {
    expect(diffWords("Email me at jane@x.com today", "Email me at <EMAIL> today")).toEqual([
      { type: "same", text: "Email me at " },
      { type: "removed", text: "jane@x.com" },
      { type: "added", text: "<EMAIL>" },
      { type: "same", text: " today" },
    ]);
  });

  it("handles identical and empty inputs", () => {
    expect(diffWords("same text", "same text")).toEqual([{ type: "same", text: "same text" }]);
    expect(diffWords("", "new")).toEqual([{ type: "added", text: "new" }]);
  });
});
