#!/usr/bin/env python3
"""Process analysis jobs from Redis queue."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.db.models import Analysis
from swellsight.db.session import SessionLocal, init_db
from swellsight.inference.batch import BatchInferenceRunner, default_checkpoint_path
from swellsight.jobs.queue import AnalysisJob, JobQueue
from swellsight.mlops.registry import load_registry
from swellsight.scoring.engine import SurfScoreEngine
from swellsight.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def process_job(job: AnalysisJob, runner: BatchInferenceRunner, scorer: SurfScoreEngine) -> bool:
    db = SessionLocal()
    row = None
    try:
        row = db.query(Analysis).filter(Analysis.id == job.analysis_id).first()
        if not row:
            logger.error("Analysis %s not found", job.analysis_id)
            return False

        row.status = "processing"
        db.commit()

        bgr = cv2.imread(job.storage_key)
        if bgr is None:
            row.status = "failed"
            row.error_message = "Could not read image"
            db.commit()
            return False

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
        logger.info("Completed analysis %s score=%s", job.analysis_id, surf_score)
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
    init_db()
    runner = BatchInferenceRunner()
    runner.warmup()
    scorer = SurfScoreEngine()
    queue = JobQueue()

    logger.info("Worker started. Checkpoint=%s", default_checkpoint_path())
    while True:
        job = queue.dequeue(timeout=5)
        if job is None:
            continue
        ok = process_job(job, runner, scorer)
        if not ok:
            queue.requeue(job)


if __name__ == "__main__":
    sys.exit(main())
