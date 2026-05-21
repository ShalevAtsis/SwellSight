"""
Redis-backed job queue for async wave analysis.
Uses a processing list for at-least-once delivery (P3-T22).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

from swellsight.platform.settings import get_settings

QUEUE_KEY = "swellsight:analysis:queue"
PROCESSING_KEY = "swellsight:analysis:processing"
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

        url = redis_url or get_settings().redis_url
        self.client = redis.from_url(url, decode_responses=True)

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
        """Move a job from queue to processing (BRPOPLPUSH)."""
        payload = self.client.brpoplpush(QUEUE_KEY, PROCESSING_KEY, timeout=timeout)
        if not payload:
            return None
        return AnalysisJob.from_json(payload)

    def ack(self, job: AnalysisJob) -> None:
        self.client.lrem(PROCESSING_KEY, 1, job.to_json())

    def requeue(self, job: AnalysisJob, delay_seconds: int = 0) -> None:
        """Return job to queue or dead-letter; non-blocking (delay handled by worker)."""
        self.client.lrem(PROCESSING_KEY, 1, job.to_json())
        job.attempts += 1
        if job.attempts >= job.max_attempts:
            self.client.rpush(DEAD_LETTER_KEY, job.to_json())
        else:
            if delay_seconds > 0:
                import time

                time.sleep(min(delay_seconds, 30))
            self.client.rpush(QUEUE_KEY, job.to_json())

    def depth(self) -> int:
        return int(self.client.llen(QUEUE_KEY))

    def processing_depth(self) -> int:
        return int(self.client.llen(PROCESSING_KEY))

    def dead_letter_depth(self) -> int:
        return int(self.client.llen(DEAD_LETTER_KEY))

    def ping(self) -> bool:
        return bool(self.client.ping())
