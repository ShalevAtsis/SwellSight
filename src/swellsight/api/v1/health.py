from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_v1():
    queue_depth = None
    try:
        from swellsight.jobs.queue import JobQueue
        queue_depth = JobQueue().depth()
    except Exception:
        pass
    return {
        "status": "ok",
        "version": "v1",
        "queue_depth": queue_depth,
    }
