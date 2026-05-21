#!/usr/bin/env python3
"""Process analysis jobs from Redis queue."""

from __future__ import annotations

import io
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.db.models import Analysis
from swellsight.db.session import SessionLocal, init_db
from swellsight.inference.batch import BatchInferenceRunner, default_checkpoint_path
from swellsight.jobs.queue import AnalysisJob, JobQueue
from swellsight.mlops.registry import load_registry
from swellsight.scoring.engine import SurfScoreEngine
from swellsight.storage import get_storage
from swellsight.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal %s received", signum)
    _shutdown = True


def load_rgb_from_storage(storage_key: str) -> np.ndarray | None:
    storage = get_storage()
    local = storage.get_local_path(storage_key)
    if local is not None:
        bgr = cv2.imread(str(local))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        raw = storage.get(storage_key)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def process_job(job: AnalysisJob, runner: BatchInferenceRunner, scorer: SurfScoreEngine) -> bool:
    db = SessionLocal()
    row = None
    try:
        row = db.query(Analysis).filter(Analysis.id == job.analysis_id).first()
        if not row:
            logger.error("Analysis %s not found", job.analysis_id)
            return True

        row.status = "processing"
        db.commit()

        rgb = load_rgb_from_storage(job.storage_key)
        if rgb is None:
            row.status = "failed"
            row.error_message = "Could not read image"
            db.commit()
            return True

        result = runner.analyze_image(rgb)
        surf_score, breakdown = scorer.compute_from_result(result)

        registry = load_registry()
        model_version = registry.active_version

        row.status = "completed"
        row.result_json = result
        row.surf_score = float(surf_score)
        row.score_breakdown = breakdown
        row.model_version = model_version
        row.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            "Completed analysis %s score=%s correlation_id=%s",
            job.analysis_id,
            surf_score,
            job.correlation_id or job.analysis_id,
        )
        return True
    except Exception as exc:
        logger.exception("Job failed: %s", exc)
        if row:
            row.status = "failed"
            row.error_message = str(exc)
            db.commit()
        return False
    finally:
        db.close()


def main() -> int:
    setup_logging()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    init_db()
    runner = BatchInferenceRunner()
    runner.warmup()
    scorer = SurfScoreEngine()
    queue = JobQueue()

    logger.info("Worker started. Checkpoint=%s", default_checkpoint_path())
    while not _shutdown:
        job = queue.dequeue(timeout=2)
        if job is None:
            continue
        ok = process_job(job, runner, scorer)
        if ok:
            queue.ack(job)
        else:
            delay = min(2 ** job.attempts, 30)
            queue.requeue(job, delay_seconds=delay)

    logger.info("Worker exiting gracefully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
