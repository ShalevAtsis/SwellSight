"""
Training callbacks for logging, checkpointing, and monitoring.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrainingCallback(ABC):
    @abstractmethod
    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_batch_start(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


class TrainingCallbacks:
    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: TrainingCallback) -> None:
        self.callbacks.append(callback)

    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)


class TensorBoardCallback(TrainingCallback):
    """Log train/val metrics to TensorBoard."""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError("Install tensorboard: pip install tensorboard") from exc
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self._phase = "train"

    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs and "phase" in logs:
            self._phase = logs["phase"]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if not logs:
            return
        prefix = logs.get("prefix", self._phase)
        for key, value in logs.items():
            if key in ("prefix", "phase"):
                continue
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{key}", value, epoch)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class JsonMetricsCallback(TrainingCallback):
    """Append per-epoch metrics to a JSON lines file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if not logs:
            return
        record = {"epoch": epoch, **logs}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def build_default_callbacks(
    log_dir: Optional[str] = None,
    tensorboard: bool = True,
) -> TrainingCallbacks:
    callbacks = TrainingCallbacks()
    if log_dir:
        callbacks.add_callback(JsonMetricsCallback(str(Path(log_dir) / "epoch_metrics.jsonl")))
        if tensorboard:
            try:
                callbacks.add_callback(TensorBoardCallback(str(Path(log_dir) / "tensorboard")))
            except ImportError:
                logger.warning("TensorBoard not installed; skipping TB callback")
    return callbacks
