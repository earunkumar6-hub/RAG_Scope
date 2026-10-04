"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

/** The run shown on a pipeline page lives in ``?run=<id>`` so past runs are linkable. */
export function useRunId(): [string | null, (id: string | null) => void] {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const setRunId = useCallback(
    (id: string | null) => router.replace(id ? `${pathname}?run=${encodeURIComponent(id)}` : pathname),
    [router, pathname],
  );
  return [params.get("run"), setRunId];
}
