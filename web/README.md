# SwellSight Web (P4)

Next.js 14 surf analysis UI — mobile-first, talks to the platform API (`/api/v1`).

## Setup

```bash
cd web
cp .env.local.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Ensure the API is running with CORS allowing the web origin:

```bash
# from repo root
export SWELLSIGHT_SKIP_MODEL_SERVER=1
export CORS_ORIGINS=http://localhost:3000
uvicorn swellsight.api.server:app --reload --port 8000
python scripts/worker.py
```

## Routes

| Path | Description |
|------|-------------|
| `/` | Landing |
| `/register`, `/login` | Auth |
| `/analyze` | Upload beach cam photo |
| `/analyze/[id]` | Poll + results (surf score, metrics, breakdown) |
| `/history` | Past analyses |

## Scripts

- `npm run dev` — development server
- `npm run build` — production build
- `npm run lint` — ESLint

## Roadmap mapping

- P4-T01–T04: App shell, design tokens, auth, API client
- P4-T05–T10: Landing, upload, polling, results, history, errors
- P4-T11: Score breakdown tooltips
