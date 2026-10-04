// Splits an answer into text and [chunk_id] citation segments (mirrors the backend's
// parse_citations: bracket groups may hold several ids separated by commas/semicolons).
export type AnswerSegment = { type: "text"; text: string } | { type: "cite"; chunkId: string };

const BRACKET = /\[([^[\]]+)\]/g;

export function splitAnswer(answer: string, validIds: Set<string>): AnswerSegment[] {
  const out: AnswerSegment[] = [];
  let last = 0;
  for (const m of answer.matchAll(BRACKET)) {
    const ids = m[1].trim().split(/[,;\s]+/);
    if (!ids.length || !ids.every((id) => validIds.has(id))) continue; // leave as plain text
    if (m.index > last) out.push({ type: "text", text: answer.slice(last, m.index) });
    ids.forEach((chunkId) => out.push({ type: "cite", chunkId }));
    last = m.index + m[0].length;
  }
  if (last < answer.length) out.push({ type: "text", text: answer.slice(last) });
  return out;
}
