# Retinal AVR — Web Frontend

Next.js (App Router) UI for the [`api/`](../api) FastAPI backend. Upload a fundus
photograph and see the full pipeline result: vessel segmentation stats, A/V
classification confidence, optic disc detection (method + confidence), and the
scientific AVR / cardiovascular risk estimate.

## Running locally

The API must be running first (see the repository root [`README.md`](../README.md)):

```bash
# from the repo root
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then, in this directory:

```bash
npm install
cp .env.local.example .env.local   # only needed if the API isn't on localhost:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Structure

- `app/page.tsx` — the single page: upload form, image preview, results display.
- `app/types.ts` — TypeScript mirror of the API's JSON response shape (kept in sync
  manually with `api/main.py`).
- `app/risk-badge.tsx` — color-coded risk level badge.

No routing, no state management library, no server-side data fetching — this is a
single client-side page calling one API endpoint.
