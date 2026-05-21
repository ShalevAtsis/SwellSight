#!/usr/bin/env python3
"""Run wave analysis inference on beach cam images."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.inference.batch import BatchInferenceRunner
from swellsight.core.pipeline import PipelineConfig
from swellsight.utils.config import ConfigManager, load_yaml_dict
from swellsight.utils.logging import setup_logging

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwellSight wave analysis inference")
    parser.add_argument("--config", default="configs/inference.yaml", help="Config YAML path")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default="outputs/inference", help="Output directory")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Trained wave model checkpoint (e.g. checkpoints/best_model.pth)",
    )
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _pipeline_config_from_yaml(yaml_path: str, checkpoint: Optional[str]) -> PipelineConfig:
    raw = load_yaml_dict(yaml_path)
    model = raw.get("model", {})
    system = raw.get("system", {})
    paths = raw.get("paths", {})
    if checkpoint:
        ckpt = checkpoint
    else:
        ckpt = paths.get("checkpoints_dir", "checkpoints")
        ckpt_path = Path(ckpt)
        if ckpt_path.is_dir():
            ckpt = str(ckpt_path / "best_model.pth")
        elif not str(ckpt).endswith(".pth"):
            ckpt = str(ckpt_path / "best_model.pth")
    backbone = model.get("backbone_model", "dinov2-base")
    backbone_map = {
        "dinov2-base": "dinov2_vitb14",
        "dinov2-small": "dinov2_vits14",
        "dinov2-large": "dinov2_vitl14",
    }
    return PipelineConfig(
        depth_model_size=model.get("depth_model_size", "base"),
        depth_precision=model.get("depth_precision", "fp16"),
        wave_backbone_model=backbone_map.get(backbone, backbone),
        freeze_backbone=model.get("freeze_backbone", True),
        use_gpu=system.get("use_gpu", True),
        max_processing_time=system.get("max_processing_time", 30.0),
        confidence_threshold=system.get("confidence_threshold", 0.7),
        wave_checkpoint_path=ckpt if ckpt and Path(ckpt).exists() else None,
        output_directory=None,
        save_intermediate_results=False,
        num_classes_breaking=model.get("num_classes_breaking", 3),
    )


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_rgb(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    args = parse_args()
    setup_logging(level="DEBUG" if args.debug else "INFO")
    logger = logging.getLogger("swellsight.inference")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        return 1

    ConfigManager(args.config)
    pipeline_config = _pipeline_config_from_yaml(args.config, args.checkpoint)
    runner = BatchInferenceRunner(config=pipeline_config)
    runner.warmup()

    images = _collect_images(input_path)
    if not images:
        logger.error("No images found at %s", input_path)
        return 1

    logger.info("Processing %s image(s)...", len(images))
    all_results = {}

    for image_path in images:
        try:
            rgb = _load_rgb(image_path)
            payload = runner.analyze_image(rgb)
            all_results[image_path.name] = payload

            out_file = output_dir / f"{image_path.stem}_analysis.json"
            with open(out_file, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

            logger.info(
                "%s -> %.2fm, %s, %s (conf %.0f%%)",
                image_path.name,
                payload["wave_height_meters"],
                payload["direction"],
                payload["breaking_type"],
                payload["overall_confidence"] * 100,
            )
        except Exception as exc:
            logger.error("Failed on %s: %s", image_path, exc, exc_info=args.debug)
            all_results[image_path.name] = {"error": str(exc)}

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    logger.info("Results saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
