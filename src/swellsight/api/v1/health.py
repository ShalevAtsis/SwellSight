from fastapi import APIRouter, HTTPException

from swellsight.platform.checks import check_database, check_redis, platform_readiness

router = APIRouter(tags=["health"])


@router.get("/health")
def health_v1():
    db_ok, db_detail = check_database()
    redis_ok, redis_detail = check_redis()
    queue_depth = None
    processing_depth = None
    if redis_ok:
        try:
            from swellsight.jobs.queue import JobQueue

            q = JobQueue()
            queue_depth = q.depth()
            processing_depth = q.processing_depth()
        except Exception:
            pass

    status = "ok" if db_ok and redis_ok else "degraded"
    return {
        "status": status,
        "version": "v1",
        "database": {"ok": db_ok, "detail": db_detail},
        "redis": {"ok": redis_ok, "detail": redis_detail},
        "queue_depth": queue_depth,
        "processing_depth": processing_depth,
    }


@router.get("/ready")
def ready_v1():
    report = platform_readiness()
    if report["status"] != "ready":
        raise HTTPException(status_code=503, detail=report)
    return report
