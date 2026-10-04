# RAGScope frontend

Next.js (App Router) + Tailwind + shadcn on Base UI, TanStack Query, Zustand, Recharts and react-force-graph. See the [root README](../README.md) for the whole project.

```bash
npm install
NEXT_PUBLIC_API_URL=http://localhost:8010 npm run dev   # PowerShell: $env:NEXT_PUBLIC_API_URL="http://localhost:8010"; npm run dev
npm test && npm run lint && npm run typecheck
npm run gen:schemas                                     # zod schemas from the backend's Pydantic models
```

- `app/`: one route per page (ingest, query, documents, graph, eval, settings)
- `components/pipeline/`: stepper, stage cards and per-stage detail views
- `components/eval/`: metrics table and charts
- `lib/`: API client, SSE hook (`useStageEvents`), generated zod schemas
- `store/`: Zustand stores

The Docker image (`Dockerfile`) is a production build with `output: "standalone"`. `NEXT_PUBLIC_API_URL` is inlined at build time.
