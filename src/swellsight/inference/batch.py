"""
Batch inference for worker and API integration.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from swellsight.core.pipeline import PipelineConfig, WaveAnalysisPipeline

logger = logging.getLogger(__name__)


def default_checkpoint_path() -> Optional[str]:
    return os.environ.get("SWELLSIGHT_CHECKPOINT") or os.environ.get("SWELLSIGHT_MODEL_PATH")


@dataclass
class BatchInferenceRunner:
    """Runs WaveAnalysisPipeline on multiple images with optional warmup."""

    config: Optional[PipelineConfig] = None
    _pipeline: Optional[WaveAnalysisPipeline] = None
    _warmed_up: bool = False

    def __post_init__(self) -> None:
        if self.config is None:
            ckpt = default_checkpoint_path()
            self.config = PipelineConfig(
                wave_checkpoint_path=ckpt,
                depth_model_size=os.environ.get("SWELLSIGHT_DEPTH_SIZE", "base"),
            )
        elif self.config.wave_checkpoint_path is None:
            self.config.wave_checkpoint_path = default_checkpoint_path()

    @property
    def pipeline(self) -> WaveAnalysisPipeline:
        if self._pipeline is None:
            self._pipeline = WaveAnalysisPipeline(config=self.config)
        return self._pipeline

    def warmup(self) -> None:
        if self._warmed_up:
            return
        logger.info("Warming up inference pipeline...")
        dummy = np.zeros((518, 518, 3), dtype=np.uint8)
        try:
            self.pipeline.process_beach_cam_image(dummy)
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): %s", exc)
        self._warmed_up = True

    def analyze_image(self, rgb: np.ndarray) -> Dict[str, Any]:
        result = self.pipeline.process_beach_cam_image(rgb)
        m = result.wave_metrics
        return {
            "wave_height_meters": float(m.height_meters),
            "wave_height_feet": float(m.height_feet),
            "direction": m.direction,
            "direction_confidence": float(m.direction_confidence),
            "breaking_type": m.breaking_type,
            "breaking_confidence": float(m.breaking_confidence),
            "overall_confidence": float(result.pipeline_confidence),
            "processing_time_seconds": float(result.processing_time),
            "warnings": list(result.warnings),
            "extreme_conditions": bool(m.extreme_conditions),
        }

    def analyze_paths(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        self.warmup()
        outputs: List[Dict[str, Any]] = []
        for path in image_paths:
            path = Path(path)
            bgr = cv2.imread(str(path))
            if bgr is None:
                outputs.append({"path": str(path), "error": "unreadable image"})
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            try:
                payload = self.analyze_image(rgb)
                payload["path"] = str(path)
                payload["latency_seconds"] = time.perf_counter() - t0
                outputs.append(payload)
            except Exception as exc:
                outputs.append({"path": str(path), "error": str(exc)})
        return outputs


def analyze_images_batch(
    image_paths: List[Union[str, Path]],
    checkpoint: Optional[str] = None,
    warmup: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience function for worker/API batch analysis."""
    config = PipelineConfig(wave_checkpoint_path=checkpoint or default_checkpoint_path())
    runner = BatchInferenceRunner(config=config)
    if warmup:
        runner.warmup()
    return runner.analyze_paths(image_paths)
