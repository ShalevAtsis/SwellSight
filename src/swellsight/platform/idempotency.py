"""Idempotency key storage (Redis) for POST /analyses (P3-T18)."""

from __future__ import annotations

import json
from typing import Optional

TTL_SECONDS = 86400


class IdempotencyStore:
    def __init__(self, redis_url: Optional[str] = None):
        import redis
        from swellsight.platform.settings import get_settings

        self.client = redis.from_url(redis_url or get_settings().redis_url, decode_responses=True)

    def _key(self, user_id: str, idempotency_key: str) -> str:
        return f"swellsight:idempotency:{user_id}:{idempotency_key}"

    def get_analysis_id(self, user_id: str, idempotency_key: str) -> Optional[str]:
        raw = self.client.get(self._key(user_id, idempotency_key))
        if not raw:
            return None
        try:
            return json.loads(raw).get("analysis_id")
        except json.JSONDecodeError:
            return raw

    def set_analysis_id(self, user_id: str, idempotency_key: str, analysis_id: str) -> bool:
        """Returns False if key already exists with different value (conflict)."""
        key = self._key(user_id, idempotency_key)
        existing = self.client.get(key)
        if existing:
            prior = self.get_analysis_id(user_id, idempotency_key)
            return prior == analysis_id
        self.client.setex(key, TTL_SECONDS, json.dumps({"analysis_id": analysis_id}))
        return True
