"""
Object storage abstraction (local filesystem or S3-compatible).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class ObjectStorage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Store bytes; returns storage key."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Read object bytes."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove object if present."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    def get_local_path(self, key: str) -> Optional[Path]:
        """Return filesystem path when backend supports direct read (local only)."""
        return None


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def get_local_path(self, key: str) -> Optional[Path]:
        path = self._path(key)
        return path if path.exists() else None


class S3ObjectStorage(ObjectStorage):
    def __init__(self, bucket: str, endpoint_url: Optional[str] = None, region: str = "us-east-1"):
        import boto3

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
        )

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def get(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


def get_storage() -> ObjectStorage:
    from swellsight.platform.settings import get_settings

    settings = get_settings()
    if settings.storage_backend == "s3":
        return S3ObjectStorage(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
        )
    return LocalObjectStorage(settings.storage_local_root)
