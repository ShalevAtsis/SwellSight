"""
Lightweight experiment logging (JSON; optional MLflow later).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ExperimentLogger:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    root_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "experiments")

    def __post_init__(self) -> None:
        self.run_dir = self.root_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.run_dir / "run.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def log_params(self, params: Dict[str, Any]) -> None:
        meta = self._load_meta()
        meta["params"] = {**meta.get("params", {}), **params}
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_meta(meta)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        record = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def log_artifact(self, source: Path, name: Optional[str] = None) -> Path:
        dest = self.run_dir / "artifacts" / (name or source.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            import shutil
            shutil.copy2(source, dest)
        return dest

    def start(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        meta = {
            "run_id": self.run_id,
            "name": name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": config or {},
            "params": {},
        }
        self._save_meta(meta)

    def _load_meta(self) -> Dict[str, Any]:
        if self.meta_path.exists():
            with open(self.meta_path, encoding="utf-8") as handle:
                return json.load(handle)
        return {"run_id": self.run_id}

    def _save_meta(self, meta: Dict[str, Any]) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
