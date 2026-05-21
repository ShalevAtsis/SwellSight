# Platform operations (P3)

Backend stack: FastAPI API, Postgres, Redis queue, GPU worker.

## Quick start (local)

```bash
pip install -e ".[platform,inference]"
docker compose -f deploy/docker-compose.platform.yml up -d postgres redis
cp .env.example .env
alembic upgrade head
export SWELLSIGHT_SKIP_MODEL_SERVER=1
uvicorn swellsight.api.server:app --reload --port 8000
python scripts/worker.py
```

## Environment

See [.env.example](../../.env.example). Required in production:

- `JWT_SECRET` — strong random value (`ENVIRONMENT=production` enforces this)
- `DATABASE_URL`, `REDIS_URL`
- `CORS_ORIGINS` — your web app origins (not `*`)

## API (v1)

| Method | Path | Auth | Notes |
|--------|------|------|-------|
| POST | `/api/v1/auth/register` | — | Password min 8 chars |
| POST | `/api/v1/auth/login` | — | Returns JWT |
| POST | `/api/v1/analyses` | Bearer | Multipart image; 202 pending |
| GET | `/api/v1/analyses/{id}` | Bearer | Poll until completed |
| GET | `/api/v1/analyses` | Bearer | History (max 100) |
| GET | `/api/v1/health` | — | DB + Redis + queue depth |
| GET | `/api/v1/ready` | — | 503 if dependencies down |

Optional header: `Idempotency-Key` on upload (24h Redis TTL).

## Hardening features

- **Storage abstraction** — `STORAGE_BACKEND=local|s3`; keys `{user_id}/{analysis_id}.ext`
- **Upload validation** — magic bytes, PIL verify, min 64px, max dimension/size
- **Rate limiting** — Redis per-IP sliding window (`RATE_LIMIT_PER_MINUTE`)
- **Daily quota** — `ANALYSES_PER_DAY_LIMIT` (default 5)
- **Queue** — `BRPOPLPUSH` to processing list; worker `ack` / retry / DLQ
- **Platform-only API** — `SWELLSIGHT_SKIP_MODEL_SERVER=1` skips sync `ModelServer` load

## Probes

| Path | Use |
|------|-----|
| `/live` | Liveness |
| `/ready` | Readiness (platform: DB+Redis; legacy: pipeline) |
| `/api/v1/health` | Versioned health detail |

## Tests

```bash
pytest tests/integration/test_platform_api.py -q
```

Uses in-memory SQLite and mocked Redis queue/idempotency.
