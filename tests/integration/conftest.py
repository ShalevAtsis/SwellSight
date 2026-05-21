"""Platform API integration test environment (SQLite, no GPU model server)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = REPO_ROOT / "data" / "test_uploads"

os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("SWELLSIGHT_SKIP_MODEL_SERVER", "1")
os.environ.setdefault("JWT_SECRET", "integration-test-secret")
os.environ.setdefault("STORAGE_LOCAL_ROOT", str(UPLOAD_ROOT))
os.environ.setdefault("ANALYSES_PER_DAY_LIMIT", "100")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

from swellsight.platform.settings import get_settings

get_settings.cache_clear()

import swellsight.db.session as db_session
from swellsight.db.models import Base
from swellsight.platform.dependencies import reset_dependencies

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
db_session.engine = _engine
db_session.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture(scope="session", autouse=True)
def _init_schema():
    Base.metadata.create_all(bind=_engine)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    yield
    reset_dependencies()
    get_settings.cache_clear()


class InMemoryJobQueue:
    def __init__(self):
        self.jobs: list = []

    def enqueue(self, analysis_id: str, user_id: str, storage_key: str):
        self.jobs.append((analysis_id, user_id, storage_key))
        from swellsight.jobs.queue import AnalysisJob
        import uuid

        return AnalysisJob(
            job_id=uuid.uuid4().hex,
            analysis_id=analysis_id,
            user_id=user_id,
            storage_key=storage_key,
        )

    def depth(self) -> int:
        return len(self.jobs)

    def ping(self) -> bool:
        return True


class InMemoryIdempotency:
    def __init__(self):
        self._store: dict[str, str] = {}

    def _key(self, user_id: str, idempotency_key: str) -> str:
        return f"{user_id}:{idempotency_key}"

    def get_analysis_id(self, user_id: str, idempotency_key: str):
        return self._store.get(self._key(user_id, idempotency_key))

    def set_analysis_id(self, user_id: str, idempotency_key: str, analysis_id: str) -> bool:
        k = self._key(user_id, idempotency_key)
        if k in self._store and self._store[k] != analysis_id:
            return False
        self._store[k] = analysis_id
        return True


@pytest.fixture
def mock_queue(monkeypatch):
    q = InMemoryJobQueue()
    monkeypatch.setattr("swellsight.api.v1.analyses.get_job_queue", lambda: q)
    monkeypatch.setattr("swellsight.jobs.queue.JobQueue", lambda *a, **k: q)
    return q


@pytest.fixture
def mock_idempotency(monkeypatch):
    store = InMemoryIdempotency()
    monkeypatch.setattr("swellsight.api.v1.analyses.get_idempotency_store", lambda: store)
    return store
