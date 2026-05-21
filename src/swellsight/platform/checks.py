"""Dependency health checks for readiness probes."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def check_database() -> Tuple[bool, str]:
    try:
        from sqlalchemy import text
        from swellsight.db.session import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def check_redis() -> Tuple[bool, str]:
    try:
        from swellsight.platform.settings import get_settings
        import redis

        client = redis.from_url(get_settings().redis_url)
        client.ping()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def platform_readiness() -> Dict[str, Any]:
    db_ok, db_msg = check_database()
    redis_ok, redis_msg = check_redis()
    queue_depth = None
    if redis_ok:
        try:
            from swellsight.jobs.queue import JobQueue

            queue_depth = JobQueue().depth()
        except Exception:
            pass

    ready = db_ok and redis_ok
    return {
        "status": "ready" if ready else "degraded",
        "database": {"ok": db_ok, "detail": db_msg},
        "redis": {"ok": redis_ok, "detail": redis_msg},
        "queue_depth": queue_depth,
    }
