import { describe, expect, it } from "vitest";

import { applyMessage, EMPTY_RUN, type RunState } from "@/lib/stages";

const stage = (stage_id: string, status: string, extra: object = {}) => ({
  run_id: "00000000-0000-0000-0000-000000000000",
  stage_id,
  status,
  started_at: "2026-10-02T00:00:00Z",
  duration_ms: null,
  params_used: {},
  summary: "",
  data: {},
  ...extra,
});

describe("applyMessage", () => {
  it("keeps the latest event per stage and appends tokens in order", () => {
    let s: RunState = EMPTY_RUN;
    s = applyMessage(s, { seq: 1, event: "stage", data: stage("Q8_generate", "pending") });
    s = applyMessage(s, { seq: 2, event: "stage", data: stage("Q8_generate", "running") });
    s = applyMessage(s, { seq: 3, event: "token", data: { text: "Hel" } });
    s = applyMessage(s, { seq: 4, event: "token", data: { text: "lo" } });
    s = applyMessage(s, { seq: 5, event: "stage", data: stage("Q8_generate", "success") });
    s = applyMessage(s, { seq: 6, event: "done", data: { status: "success", answer: "Hello" } });
    expect(s.stages.Q8_generate?.status).toBe("success");
    expect(s.tokens).toBe("Hello");
    expect(s.done?.status).toBe("success");
    expect(s.lastSeq).toBe(6);
  });

  it("ignores replayed messages at or below lastSeq (reconnect overlap)", () => {
    let s: RunState = EMPTY_RUN;
    s = applyMessage(s, { seq: 1, event: "token", data: { text: "a" } });
    s = applyMessage(s, { seq: 2, event: "token", data: { text: "b" } });
    s = applyMessage(s, { seq: 2, event: "token", data: { text: "b" } });
    s = applyMessage(s, { seq: 1, event: "token", data: { text: "a" } });
    expect(s.tokens).toBe("ab");
  });

  it("keeps the final answer that arrives before Q10 finishes", () => {
    let s: RunState = EMPTY_RUN;
    s = applyMessage(s, { seq: 1, event: "answer", data: { status: "success", answer: "Final" } });
    s = applyMessage(s, { seq: 2, event: "stage", data: stage("Q10_eval", "running") });
    expect(s.answer?.answer).toBe("Final");
    expect(s.done).toBeNull();
  });
});
