"use client";

export const td = "px-2 py-1 align-top";

export function Table({ head, children }: { head: string[]; children: React.ReactNode }) {
  return (
    <div className="max-h-96 overflow-auto rounded border">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-muted text-left text-muted-foreground">
          <tr>{head.map((h) => <th key={h} className="px-2 py-1 font-medium">{h}</th>)}</tr>
        </thead>
        <tbody className="divide-y">{children}</tbody>
      </table>
    </div>
  );
}
