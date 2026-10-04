import { Suspense } from "react";

import { QueryView } from "@/app/query/QueryView";

export default function QueryPage() {
  // QueryView reads ?run= (useSearchParams), which needs a Suspense boundary when prerendering.
  return (
    <Suspense>
      <QueryView />
    </Suspense>
  );
}
