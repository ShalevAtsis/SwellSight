"""
Environment-driven platform settings (P3).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional


class PlatformSettings:
    def __init__(self) -> None:
        self.database_url: str = os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg2://swellsight:swellsight@localhost:5432/swellsight",
        )
        self.redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.jwt_secret: str = os.environ.get("JWT_SECRET", "change-me-in-production")
        self.jwt_expire_minutes: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

        self.storage_backend: str = os.environ.get("STORAGE_BACKEND", "local")  # local | s3
        self.storage_local_root: str = os.environ.get("STORAGE_LOCAL_ROOT", "data/uploads")
        self.s3_bucket: str = os.environ.get("S3_BUCKET", "swellsight-uploads")
        self.s3_endpoint_url: Optional[str] = os.environ.get("S3_ENDPOINT_URL")
        self.s3_region: str = os.environ.get("S3_REGION", "us-east-1")

        self.cors_origins: List[str] = [
            o.strip()
            for o in os.environ.get(
                "CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if o.strip()
        ]

        self.max_upload_bytes: int = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
        self.max_image_dimension: int = int(os.environ.get("MAX_IMAGE_DIMENSION", "4096"))
        self.analyses_per_day_limit: int = int(os.environ.get("ANALYSES_PER_DAY_LIMIT", "5"))
        self.rate_limit_per_minute: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

        self.skip_model_server: bool = os.environ.get("SWELLSIGHT_SKIP_MODEL_SERVER", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self.require_secure_jwt: bool = os.environ.get("ENVIRONMENT", "development") == "production"

        self.swellsight_checkpoint: Optional[str] = os.environ.get("SWELLSIGHT_CHECKPOINT")

    def validate_startup(self) -> None:
        if self.require_secure_jwt and self.jwt_secret in (
            "change-me-in-production",
            "dev-secret-change-in-prod",
        ):
            raise RuntimeError("JWT_SECRET must be set in production")


@lru_cache
def get_settings() -> PlatformSettings:
    return PlatformSettings()
