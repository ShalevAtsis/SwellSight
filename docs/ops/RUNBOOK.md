# SwellSight operations runbook (P5-T20)

Deploy, rollback, and day-2 operations for the platform stack.

## Architecture

| Service | Image / process | Port |
|---------|-----------------|------|
| API | `deploy/Dockerfile.api` or `uvicorn` | 8000 |
| Worker | `deploy/Dockerfile.worker` or `scripts/worker.py` | — |
| Postgres | `postgres:15-alpine` | 5432 |
| Redis | `redis:7-alpine` | 6379 |
| Web | Next.js (`web/`) | 3000 |

## Deploy (Docker Compose)

From repo root:

```powershell
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml run --rm migrate
docker compose -f deploy/docker-compose.yml up -d postgres redis api worker
```

Place model weights at `checkpoints/best_model.pth` on the host (mounted read-only into worker).

Verify:

- http://localhost:8000/api/v1/health — `database.ok` and `redis.ok` true
- http://localhost:8000/docs — OpenAPI
- Worker logs: `Worker started`

## Rollback

1. Stop current stack: `docker compose -f deploy/docker-compose.yml down`
2. Check out previous git tag or image digest.
3. Rebuild/repull images and `up` again.
4. DB migrations are forward-only; restore Postgres snapshot if a bad migration shipped.

## Promote a new model

1. Train and evaluate locally ([MODEL_GUIDE.md](../MODEL_GUIDE.md)).
2. `python scripts/promote_model.py --version wave-vX.Y.Z --checkpoint checkpoints/best_model.pth`
3. Set worker env `SWELLSIGHT_CHECKPOINT` to promoted path.
4. Restart worker only (API unchanged if `SWELLSIGHT_SKIP_MODEL_SERVER=1`).

## Common incidents

| Symptom | Check | Fix |
|---------|-------|-----|
| Upload stays pending | Worker running? Redis up? | Start worker; `redis-cli ping` |
| 503 on `/ready` | Postgres/Redis | `docker compose ps`; fix `DATABASE_URL` |
| All analyses failed | Worker logs, checkpoint | Mount `best_model.pth`; GPU drivers |
| 429 on upload | Daily limit | Raise `ANALYSES_PER_DAY_LIMIT` for dev |
| CORS errors from web | API env | `CORS_ORIGINS=http://localhost:3000` |

## Secrets (production)

Rotate periodically (P5-T12):

- `JWT_SECRET` — invalidate all sessions on change
- `DATABASE_URL` — Postgres credentials
- Hugging Face token — depth/model download on worker
- S3/MinIO keys — when `STORAGE_BACKEND=s3`

## Image retention (7 days default)

```powershell
# Dry run
python scripts/cleanup_uploads.py --dry-run

# Delete files older than UPLOAD_RETENTION_DAYS (default 7)
python scripts/cleanup_uploads.py
```

Schedule via cron or Kubernetes CronJob in production.

## Metrics

- Prometheus scrape: `GET http://api:8000/metrics`
- Gauges: `swellsight_queue_depth`, `swellsight_queue_processing`, `swellsight_queue_dead_letter`

## Logs

- API/worker: stdout JSON/text via Docker or process manager
- Worker logs include `correlation_id` (= analysis id)
- Queue depth: `GET /api/v1/health` → `queue_depth`
- Dead letter: Redis key `swellsight:analysis:dead`

## Related

- [RUN_LOCALLY.md](../RUN_LOCALLY.md) — developer setup
- [PLATFORM.md](PLATFORM.md) — API and env reference
