# SwellSight Web (P4)

Next.js 14 surf analysis UI — mobile-first, talks to the platform API (`/api/v1`).

## Setup

**Full stack (API, Docker, worker, web):** see [docs/RUN_LOCALLY.md](../docs/RUN_LOCALLY.md).

```bash
cd web
cp .env.local.example .env.local   # Windows: Copy-Item .env.local.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The API and worker must be running — see RUN_LOCALLY.md for PowerShell commands (`$env:...` and `python -m uvicorn`).

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
