"""Shared platform singletons."""

from __future__ import annotations

from typing import Optional

from swellsight.jobs.queue import JobQueue
from swellsight.platform.idempotency import IdempotencyStore

_queue: Optional[JobQueue] = None
_idempotency: Optional[IdempotencyStore] = None


def get_job_queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue


def get_idempotency_store() -> IdempotencyStore:
    global _idempotency
    if _idempotency is None:
        _idempotency = IdempotencyStore()
    return _idempotency


def reset_dependencies() -> None:
    """Test helper to clear cached clients."""
    global _queue, _idempotency
    _queue = None
    _idempotency = None
