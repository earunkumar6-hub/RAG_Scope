import { describe, expect, it } from "vitest";

import { splitAnswer } from "@/lib/citations";

describe("splitAnswer", () => {
  const valid = new Set(["0a1b2c3d-v1-0", "0a1b2c3d-v1-1"]);

  it("turns known [chunk_id] groups into citations and keeps other brackets as text", () => {
    const segs = splitAnswer("A [0a1b2c3d-v1-1]. B [0a1b2c3d-v1-0, 0a1b2c3d-v1-1] [note] [ffffffff-v1-9]", valid);
    expect(segs).toEqual([
      { type: "text", text: "A " },
      { type: "cite", chunkId: "0a1b2c3d-v1-1" },
      { type: "text", text: ". B " },
      { type: "cite", chunkId: "0a1b2c3d-v1-0" },
      { type: "cite", chunkId: "0a1b2c3d-v1-1" },
      { type: "text", text: " [note] [ffffffff-v1-9]" },
    ]);
  });

  it("handles answers without citations", () => {
    expect(splitAnswer("plain", valid)).toEqual([{ type: "text", text: "plain" }]);
  });
});
