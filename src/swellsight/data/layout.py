"""
Standard dataset directory layout for SwellSight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

STANDARD_DIRS = [
    "data/raw/beach_cams",
    "data/processed",
    "data/depth_maps",
    "data/synthetic",
    "data/augmented",
    "data/manifests",
    "checkpoints",
    "outputs/training",
    "outputs/inference",
    "outputs/evaluation",
    "logs",
]


@dataclass
class LayoutReport:
    missing: List[str]
    existing: List[str]

    @property
    def ok(self) -> bool:
        return len(self.missing) == 0


def resolve_data_root(data_root: Optional[str | Path] = None) -> Path:
    if data_root is None:
        return REPO_ROOT / "data"
    return Path(data_root)


def validate_layout(
    data_root: Optional[str | Path] = None,
    create_missing: bool = False,
) -> LayoutReport:
    """Check standard directories exist under repo (or data_root parent)."""
    root = REPO_ROOT if data_root is None else Path(data_root).parent
    missing: List[str] = []
    existing: List[str] = []

    for rel in STANDARD_DIRS:
        path = root / rel
        if path.exists():
            existing.append(rel)
        else:
            missing.append(rel)
            if create_missing:
                path.mkdir(parents=True, exist_ok=True)
                existing.append(rel)
                missing.remove(rel)

    return LayoutReport(missing=missing, existing=existing)


def layout_help() -> str:
    lines = ["SwellSight standard data layout:", ""]
    for rel in STANDARD_DIRS:
        lines.append(f"  {rel}/")
    lines.extend(
        [
            "",
            "Training .npy pairs (in data/ or data/synthetic/):",
            "  sample_001.npy",
            "  sample_001_labels.npy",
            "  sample_001_depth.npy  (optional, from extract_depth_maps.py)",
        ]
    )
    return "\n".join(lines)
