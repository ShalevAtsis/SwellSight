#!/usr/bin/env python3
"""Delete upload images older than retention policy (P5-T13)."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.db.models import Analysis
from swellsight.db.session import SessionLocal, init_db
from swellsight.platform.settings import get_settings
from swellsight.storage import get_storage
from swellsight.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Purge uploads past retention period")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Retention days (default: UPLOAD_RETENTION_DAYS env or 7)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log only, do not delete")
    args = parser.parse_args()

    import os

    days = args.days or int(os.environ.get("UPLOAD_RETENTION_DAYS", "7"))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    init_db()
    storage = get_storage()
    db = SessionLocal()
    deleted = 0

    try:
        rows = (
            db.query(Analysis)
            .filter(Analysis.created_at < cutoff, Analysis.storage_key.isnot(None))
            .all()
        )
        for row in rows:
            key = row.storage_key
            if not key:
                continue
            if args.dry_run:
                logger.info("Would delete %s (analysis %s)", key, row.id)
            else:
                try:
                    storage.delete(key)
                    deleted += 1
                    logger.info("Deleted %s", key)
                except Exception as exc:
                    logger.warning("Failed to delete %s: %s", key, exc)
        logger.info(
            "Retention %s days: %s files %s",
            days,
            len(rows),
            "would delete" if args.dry_run else f"deleted {deleted}",
        )
    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
