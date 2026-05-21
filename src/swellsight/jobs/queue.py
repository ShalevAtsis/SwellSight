"""
Redis-backed job queue for async wave analysis.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = "swellsight:analysis:queue"
DEAD_LETTER_KEY = "swellsight:analysis:dead"


@dataclass
class AnalysisJob:
    job_id: str
    analysis_id: str
    user_id: str
    storage_key: str
    attempts: int = 0
    max_attempts: int = 3

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "AnalysisJob":
        return cls(**json.loads(raw))


class JobQueue:
    def __init__(self, redis_url: Optional[str] = None):
        import redis

        self.client = redis.from_url(redis_url or REDIS_URL, decode_responses=True)

    def enqueue(self, analysis_id: str, user_id: str, storage_key: str) -> AnalysisJob:
        job = AnalysisJob(
            job_id=uuid.uuid4().hex,
            analysis_id=analysis_id,
            user_id=user_id,
            storage_key=storage_key,
        )
        self.client.rpush(QUEUE_KEY, job.to_json())
        return job

    def dequeue(self, timeout: int = 5) -> Optional[AnalysisJob]:
        item = self.client.blpop(QUEUE_KEY, timeout=timeout)
        if not item:
            return None
        _, payload = item
        return AnalysisJob.from_json(payload)

    def requeue(self, job: AnalysisJob) -> None:
        job.attempts += 1
        if job.attempts >= job.max_attempts:
            self.client.rpush(DEAD_LETTER_KEY, job.to_json())
        else:
            time.sleep(min(2 ** job.attempts, 30))
            self.client.rpush(QUEUE_KEY, job.to_json())

    def depth(self) -> int:
        return int(self.client.llen(QUEUE_KEY))
