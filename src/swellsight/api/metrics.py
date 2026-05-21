"""Prometheus-style metrics for platform monitoring (P5-T17)."""

from __future__ import annotations

import time
from typing import Callable

from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

router = APIRouter(tags=["metrics"])

_request_count = 0
_request_errors = 0
_latency_sum = 0.0


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        global _request_count, _request_errors, _latency_sum
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        _request_count += 1
        try:
            response = await call_next(request)
            if response.status_code >= 500:
                _request_errors += 1
            return response
        except Exception:
            _request_errors += 1
            raise
        finally:
            _latency_sum += time.perf_counter() - start


@router.get("/metrics")
def prometheus_metrics():
    """Plain-text metrics for Prometheus scrape."""
    lines = [
        "# HELP swellsight_http_requests_total Total HTTP requests",
        "# TYPE swellsight_http_requests_total counter",
        f"swellsight_http_requests_total {_request_count}",
        "# HELP swellsight_http_errors_total HTTP 5xx responses",
        "# TYPE swellsight_http_errors_total counter",
        f"swellsight_http_errors_total {_request_errors}",
        "# HELP swellsight_http_latency_seconds_sum Sum of request latencies",
        "# TYPE swellsight_http_latency_seconds_sum counter",
        f"swellsight_http_latency_seconds_sum {_latency_sum:.6f}",
    ]

    try:
        from swellsight.jobs.queue import JobQueue

        q = JobQueue()
        lines.extend(
            [
                "# HELP swellsight_queue_depth Pending jobs",
                "# TYPE swellsight_queue_depth gauge",
                f"swellsight_queue_depth {q.depth()}",
                "# HELP swellsight_queue_processing In-flight jobs",
                "# TYPE swellsight_queue_processing gauge",
                f"swellsight_queue_processing {q.processing_depth()}",
                "# HELP swellsight_queue_dead_letter Dead letter jobs",
                "# TYPE swellsight_queue_dead_letter gauge",
                f"swellsight_queue_dead_letter {q.dead_letter_depth()}",
            ]
        )
    except Exception:
        pass

    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
