# Run SwellSight locally (A–Z)

End-to-end guide to run the **web app**, **API**, **database**, **queue**, and **GPU worker** on your machine.

**Time:** ~30 minutes first time (excluding model download/training).  
**OS:** Windows (PowerShell) primary; macOS/Linux notes where different.

---

## What you are starting

```mermaid
flowchart LR
  Browser["Browser :3000"] --> Web["Next.js web"]
  Web --> API["FastAPI :8000"]
  API --> PG[(Postgres)]
  API --> Redis[(Redis)]
  API -->|enqueue job| Redis
  Worker["worker.py GPU"] --> Redis
  Worker --> PG
  Worker --> Storage["data/uploads"]
```

| Component | Port | Role |
|-----------|------|------|
| **Web** (`web/`) | 3000 | Login, upload photo, view surf score |
| **API** | 8000 | Auth, analyses, health |
| **Postgres** | 5432 | Users, analysis records |
| **Redis** | 6379 | Job queue, rate limit, idempotency |
| **Worker** | — | Picks jobs, runs wave model, writes results |

You need **four processes** for the full product path: Postgres, Redis, API, worker — plus the web dev server.

---

## Prerequisites

Install before you start:

| Tool | Version | Check |
|------|---------|--------|
| **Python** | 3.10 or 3.11 recommended | `python --version` |
| **Node.js** | 18+ | `node --version` |
| **Docker Desktop** | Latest (for Postgres + Redis) | `docker --version` |
| **Git** | Any | `git --version` |

**For the worker (wave analysis):**

- NVIDIA GPU with drivers (recommended). CPU may work but is very slow.
- A trained checkpoint at `checkpoints/best_model.pth` **or** set `SWELLSIGHT_CHECKPOINT` to your `.pth` file.  
  Without a checkpoint, jobs may fail or return poor results. See [MODEL_GUIDE.md](MODEL_GUIDE.md) / [TRAINING_FROM_SCRATCH.md](TRAINING_FROM_SCRATCH.md) to train one.

---

## Step 1 — Clone and open the repo

```powershell
cd C:\dev\ai\SwellSight
```

(Adjust path if you cloned elsewhere.)

---

## Step 2 — Python virtual environment and install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the package with API + platform extras:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[inference,platform]"
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[inference,platform]"
```

---

## Step 3 — Environment file

Copy the example env file (optional but recommended):

```powershell
Copy-Item .env.example .env
```

Edit `.env` if needed. For local dev these values are fine:

| Variable | Local value |
|----------|-------------|
| `DATABASE_URL` | `postgresql+psycopg2://swellsight:swellsight@localhost:5432/swellsight` |
| `REDIS_URL` | `redis://localhost:6379/0` |
| `SWELLSIGHT_SKIP_MODEL_SERVER` | `1` (API does not load heavy ML on startup) |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` |
| `STORAGE_LOCAL_ROOT` | `data/uploads` |
| `SWELLSIGHT_CHECKPOINT` | `checkpoints/best_model.pth` (if you have one) |

**PowerShell (current session)** — use this every time you open a new terminal unless you use `.env` with a loader:

```powershell
$env:DATABASE_URL = "postgresql+psycopg2://swellsight:swellsight@localhost:5432/swellsight"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:SWELLSIGHT_SKIP_MODEL_SERVER = "1"
$env:CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
$env:JWT_SECRET = "dev-local-secret"
$env:STORAGE_LOCAL_ROOT = "data\uploads"
```

> **Note:** Python does not auto-load `.env` unless you use a tool like `python-dotenv`. Setting variables in PowerShell (above) or exporting them in bash is the reliable approach for local runs.

---

## Step 4 — Start Postgres and Redis (Docker)

From the repo root:

```powershell
docker compose -f deploy/docker-compose.platform.yml up -d postgres redis
```

Wait until both are healthy:

```powershell
docker compose -f deploy/docker-compose.platform.yml ps
```

You should see `postgres` and `redis` running.

---

## Step 5 — Database migrations

With the venv activated and `DATABASE_URL` set:

```powershell
python -m alembic upgrade head
```

If that fails, try:

```powershell
alembic upgrade head
```

Creates tables: `users`, `analyses`, `model_versions`.

---

## Step 6 — Web app setup (one-time)

```powershell
cd web
Copy-Item .env.local.example .env.local
npm install
cd ..
```

`.env.local` should contain:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Step 7 — Run the system (four terminals)

Activate the venv and set env vars in **each** Python terminal (Step 3).

### Terminal A — API server

```powershell
cd C:\dev\ai\SwellSight
.\.venv\Scripts\Activate.ps1
$env:SWELLSIGHT_SKIP_MODEL_SERVER = "1"
$env:CORS_ORIGINS = "http://localhost:3000"
$env:DATABASE_URL = "postgresql+psycopg2://swellsight:swellsight@localhost:5432/swellsight"
$env:REDIS_URL = "redis://localhost:6379/0"
python -m uvicorn swellsight.api.server:app --reload --port 8000
```

**Verify:** open http://127.0.0.1:8000/docs (Swagger UI).

### Terminal B — Worker (processes uploads)

```powershell
cd C:\dev\ai\SwellSight
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL = "postgresql+psycopg2://swellsight:swellsight@localhost:5432/swellsight"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:SWELLSIGHT_CHECKPOINT = "checkpoints\best_model.pth"
python scripts/worker.py
```

Leave this running. First startup may download depth/wave models (slow).

### Terminal C — Web UI

```powershell
cd C:\dev\ai\SwellSight\web
npm run dev
```

**Verify:** open http://localhost:3000

### Terminal D — Docker (only if not already up)

If you stopped Postgres/Redis, rerun Step 4 in this terminal.

---

## Step 8 — Full user journey (smoke test)

1. Open **http://localhost:3000**
2. Click **Sign up** → email + password (min 8 characters)
3. Go to **Analyze** → upload a beach/wave image (JPEG/PNG/WebP, max 10 MB)
4. You are redirected to the analysis page — status **pending** → **processing** → **completed**
5. View **surf score** (0–100), wave metrics, and score breakdown
6. Open **History** to see past runs

**API-only check (optional):**

```powershell
# Register
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/auth/register -Method POST -ContentType "application/json" -Body '{"email":"test@example.com","password":"password123"}'

# Health
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/health
```

---

## Quick reference — URLs

| URL | Purpose |
|-----|---------|
| http://localhost:3000 | Web app |
| http://127.0.0.1:8000/docs | API Swagger |
| http://127.0.0.1:8000/api/v1/health | DB + Redis health |
| http://127.0.0.1:8000/api/v1/ready | Readiness probe |

---

## Stopping everything

```powershell
# Ctrl+C in API, worker, and web terminals

docker compose -f deploy/docker-compose.platform.yml down
```

Data in Postgres persists in Docker volume `pgdata` until you remove volumes.

---

## Troubleshooting

### `uvicorn` is not recognized

Use the module form (venv activated):

```powershell
python -m uvicorn swellsight.api.server:app --reload --port 8000
```

### `export VAR=value` fails on PowerShell

Use `$env:VAR = "value"` instead of `export`.

### `ModuleNotFoundError: No module named 'swellsight'`

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[inference,platform]"
```

### API starts but register/upload fails (database)

- Is Docker Postgres running? `docker compose -f deploy/docker-compose.platform.yml ps`
- Did you run `alembic upgrade head`?
- Is `DATABASE_URL` correct?

### Upload stays **pending** forever

- Is the **worker** running (Terminal B)?
- Is **Redis** up? Check http://127.0.0.1:8000/api/v1/health → `redis.ok` should be true.
- Worker logs may show missing checkpoint or GPU errors.

### CORS error in browser

Set on the API process:

```powershell
$env:CORS_ORIGINS = "http://localhost:3000"
```

Restart the API after changing.

### Analysis **failed** in the UI

- Read `error_message` on the result page.
- Common causes: corrupt image, no checkpoint, worker OOM, depth model download failure.
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for ML-specific issues.

### Daily limit (429)

Default is 5 analyses per user per day. For local testing:

```powershell
$env:ANALYSES_PER_DAY_LIMIT = "100"
```

Restart the API.

---

## Optional paths

| Goal | Doc |
|------|-----|
| Train a model from scratch | [TRAINING_FROM_SCRATCH.md](TRAINING_FROM_SCRATCH.md) |
| Model / checkpoint usage | [MODEL_GUIDE.md](MODEL_GUIDE.md) |
| Platform API details | [ops/PLATFORM.md](ops/PLATFORM.md) |
| Product roadmap | [SYSTEM_ROADMAP.md](SYSTEM_ROADMAP.md) |
| Run integration tests (no Docker) | `pytest tests/integration/test_platform_api.py -q` |

---

## Architecture checklist

Before demoing, confirm:

- [ ] `docker compose ... ps` → postgres + redis **Up**
- [ ] `python -m alembic upgrade head` succeeded
- [ ] API: http://127.0.0.1:8000/api/v1/health → `"status": "ok"` (or degraded only if Redis/DB down)
- [ ] Worker terminal shows `Worker started`
- [ ] Web: http://localhost:3000 loads
- [ ] Checkpoint exists (if you expect real scores): `checkpoints/best_model.pth`

You now have the full SwellSight stack running locally.
