"use client";

// Subscribes to a run's SSE stream and folds it into RunState. The browser's EventSource resends
// Last-Event-ID on reconnect, and the backend replays from there (or from SQLite for old runs).
import { useEffect, useReducer } from "react";

import { eventsUrl } from "@/lib/api";
import { applyMessage, EMPTY_RUN, type RunKind, type RunState, type SseMessage } from "@/lib/stages";

export type Connection = "idle" | "connecting" | "open" | "closed" | "error";

// State is tagged with the run it belongs to: switching runs needs no reset inside the effect,
// and a late message from a previous run's stream cannot leak into the new one.
type Tagged = { key: string | null; run: RunState; connection: Connection };
type Action =
  | { key: string; type: "message"; msg: SseMessage }
  | { key: string; type: "connection"; value: Connection };

function reducer(state: Tagged, action: Action): Tagged {
  const base = state.key === action.key ? state : { key: action.key, run: EMPTY_RUN, connection: "connecting" as const };
  if (action.type === "connection") {
    // Once "done" closed the stream, a late error/retry callback must not override it.
    return base.connection === "closed" ? base : { ...base, connection: action.value };
  }
  const run = applyMessage(base.run, action.msg);
  return { ...base, run, connection: action.msg.event === "done" ? "closed" : base.connection };
}

export function useStageEvents(kind: RunKind, runId: string | null) {
  const [state, dispatch] = useReducer(reducer, { key: null, run: EMPTY_RUN, connection: "idle" });

  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(eventsUrl(kind, runId));
    const handle = (event: SseMessage["event"]) => (e: MessageEvent<string>) => {
      dispatch({ key: runId, type: "message", msg: { seq: Number(e.lastEventId) || 0, event, data: JSON.parse(e.data) } });
      if (event === "done") source.close();
    };
    source.onopen = () => dispatch({ key: runId, type: "connection", value: "open" });
    source.addEventListener("stage", handle("stage"));
    source.addEventListener("token", handle("token"));
    source.addEventListener("answer", handle("answer"));
    source.addEventListener("done", handle("done"));
    source.onerror = () =>
      // CLOSED: the server refused (404 unknown run / 204 already finished); otherwise the
      // browser is retrying on its own.
      dispatch({ key: runId, type: "connection", value: source.readyState === EventSource.CLOSED ? "error" : "connecting" });
    return () => source.close();
  }, [kind, runId]);

  if (!runId) return { state: EMPTY_RUN, connection: "idle" as Connection };
  if (state.key !== runId) return { state: EMPTY_RUN, connection: "connecting" as Connection };
  return { state: state.run, connection: state.connection };
}
