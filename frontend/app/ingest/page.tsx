import { Suspense } from "react";

import { IngestView } from "@/app/ingest/IngestView";

export default function IngestPage() {
  // IngestView reads ?run= (useSearchParams), which needs a Suspense boundary when prerendering.
  return (
    <Suspense>
      <IngestView />
    </Suspense>
  );
}
