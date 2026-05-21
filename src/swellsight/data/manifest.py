"""
Build and load dataset manifests for reproducible training.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

IMAGE_SUFFIXES = (".npy",)
LABEL_SUFFIX = "_labels.npy"
DEPTH_SUFFIX = "_depth.npy"


@dataclass
class ManifestEntry:
    id: str
    image_path: str
    label_path: str
    depth_path: Optional[str] = None
    split: str = "train"


@dataclass
class DatasetManifest:
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_dir: str = "data"
    train_ratio: float = 0.8
    entries: List[ManifestEntry] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "data_dir": self.data_dir,
            "train_ratio": self.train_ratio,
            "entries": [asdict(e) for e in self.entries],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        entries = [ManifestEntry(**e) for e in data.get("entries", [])]
        return cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            data_dir=data.get("data_dir", "data"),
            train_ratio=data.get("train_ratio", 0.8),
            entries=entries,
            stats=data.get("stats", {}),
        )


def _file_hash(path: Path, max_bytes: int = 1_000_000) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:16]


def build_manifest(
    data_dir: str | Path,
    output_path: Optional[str | Path] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetManifest:
    """Scan data_dir for .npy training pairs and write manifest JSON."""
    import random

    data_path = Path(data_dir)
    output_path = Path(output_path or data_path.parent / "manifests" / "dataset_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f
        for f in data_path.glob("*.npy")
        if LABEL_SUFFIX not in f.name and DEPTH_SUFFIX not in f.name
    )

    entries: List[ManifestEntry] = []
    for img_file in image_files:
        label_file = data_path / f"{img_file.stem}{LABEL_SUFFIX}"
        if not label_file.exists():
            continue
        depth_file = data_path / f"{img_file.stem}{DEPTH_SUFFIX}"
        entries.append(
            ManifestEntry(
                id=img_file.stem,
                image_path=str(img_file.name),
                label_path=str(label_file.name),
                depth_path=str(depth_file.name) if depth_file.exists() else None,
            )
        )

    random.Random(seed).shuffle(entries)
    split_idx = int(len(entries) * train_ratio)
    for i, entry in enumerate(entries):
        entry.split = "train" if i < split_idx else "validation"

    manifest = DatasetManifest(
        data_dir=str(data_path),
        train_ratio=train_ratio,
        entries=entries,
        stats={
            "total": len(entries),
            "train": sum(1 for e in entries if e.split == "train"),
            "validation": sum(1 for e in entries if e.split == "validation"),
            "with_depth": sum(1 for e in entries if e.depth_path),
            "manifest_hash": hashlib.sha256(json.dumps([e.id for e in entries]).encode()).hexdigest()[:16],
        },
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)

    return manifest


def load_manifest(path: str | Path) -> DatasetManifest:
    with open(path, encoding="utf-8") as handle:
        return DatasetManifest.from_dict(json.load(handle))
