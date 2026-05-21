"""
Redis sliding-window rate limiting per client IP (P3-T09).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: Optional[str] = None, limit_per_minute: int = 60):
        super().__init__(app)
        self.limit_per_minute = limit_per_minute
        self._redis = None
        self._redis_url = redis_url

    def _client(self):
        if self._redis is None:
            import redis
            from swellsight.platform.settings import get_settings

            self._redis = redis.from_url(
                self._redis_url or get_settings().redis_url,
                decode_responses=True,
            )
        return self._redis

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path.endswith("/health") or request.url.path.endswith("/live") or request.url.path == "/metrics":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = f"swellsight:ratelimit:{client_ip}:{int(time.time()) // 60}"

        try:
            count = self._client().incr(key)
            if count == 1:
                self._client().expire(key, 120)
            if count > self.limit_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )
        except Exception:
            pass

        return await call_next(request)
