"""
Model registry for checkpoint promotion and deployment tracking.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = REPO_ROOT / "models" / "registry.yaml"
PROMOTED_DIR = REPO_ROOT / "models" / "promoted"


@dataclass
class ModelVersion:
    id: str
    checkpoint: str
    created_at: str
    git_sha: Optional[str] = None
    data_manifest: Optional[str] = None
    data_manifest_sha256: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "staging"
    model_card: Optional[str] = None


@dataclass
class ModelRegistry:
    active_version: Optional[str]
    versions: List[ModelVersion]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRegistry":
        versions = [ModelVersion(**v) for v in data.get("versions") or []]
        return cls(active_version=data.get("active_version"), versions=versions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_version": self.active_version,
            "versions": [
                {
                    "id": v.id,
                    "checkpoint": v.checkpoint,
                    "created_at": v.created_at,
                    "git_sha": v.git_sha,
                    "data_manifest": v.data_manifest,
                    "data_manifest_sha256": v.data_manifest_sha256,
                    "metrics": v.metrics,
                    "status": v.status,
                    "model_card": v.model_card,
                }
                for v in self.versions
            ],
        }


def load_registry(path: Optional[Path] = None) -> ModelRegistry:
    path = path or DEFAULT_REGISTRY
    if not path.exists():
        return ModelRegistry(active_version=None, versions=[])
    with open(path, encoding="utf-8") as handle:
        return ModelRegistry.from_dict(yaml.safe_load(handle) or {})


def save_registry(registry: ModelRegistry, path: Optional[Path] = None) -> None:
    path = path or DEFAULT_REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(registry.to_dict(), handle, default_flow_style=False, sort_keys=False)


def _file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> Optional[str]:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def promote_checkpoint(
    version_id: str,
    checkpoint_path: str | Path,
    metrics: Optional[Dict[str, Any]] = None,
    data_manifest: Optional[str] = None,
    status: str = "staging",
    set_active: bool = True,
) -> ModelVersion:
    """Copy checkpoint to models/promoted/ and update registry."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROMOTED_DIR / f"{version_id}.pth"
    shutil.copy2(checkpoint_path, dest)
    try:
        ckpt_ref = str(dest.relative_to(REPO_ROOT))
    except ValueError:
        ckpt_ref = str(dest)

    manifest_sha = None
    if data_manifest:
        manifest_path = Path(data_manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest_sha = _file_sha256(manifest_path)

    version = ModelVersion(
        id=version_id,
        checkpoint=ckpt_ref,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(),
        data_manifest=data_manifest,
        data_manifest_sha256=manifest_sha,
        metrics=metrics or {},
        status=status,
        model_card=f"docs/models/{version_id}.md",
    )

    registry = load_registry()
    registry.versions = [v for v in registry.versions if v.id != version_id]
    registry.versions.append(version)
    if set_active:
        registry.active_version = version_id

    save_registry(registry)
    return version
