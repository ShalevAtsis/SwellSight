#!/usr/bin/env python3
"""Generate synthetic wave training images (local; no Colab)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.core.depth_extractor import DepthMap
from swellsight.core.synthetic_generator import (
    FLUXControlNetGenerator,
    GenerationConfig,
    WeatherConditions,
    create_default_weather_conditions,
)
from swellsight.utils.config import load_yaml_dict
from swellsight.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic SwellSight training data")
    parser.add_argument("--config", default="configs/synthetic.yaml", help="Synthetic job config")
    parser.add_argument("--depth-dir", default="data/depth_maps", help="Source depth maps")
    parser.add_argument("--output", default="data/synthetic", help="Output directory")
    parser.add_argument("--num-images", type=int, default=None, help="Override config count")
    parser.add_argument(
        "--mode",
        choices=["from-depth", "balanced"],
        default="from-depth",
        help="from-depth: one image per depth file; balanced: FLUX balanced generator",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max images (testing)")
    parser.add_argument("--dry-run", action="store_true", help="List work only, no GPU generation")
    return parser.parse_args()


def setup_hf_auth() -> bool:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set; gated FLUX models may fail to download")
        return False
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    except Exception as exc:
        logger.warning("HF login failed: %s", exc)
        return False


def load_depth_map(path: Path) -> DepthMap:
    data = np.load(path).astype(np.float32)
    if data.ndim == 3:
        data = data.squeeze()
    dmax = float(data.max()) if data.size else 1.0
    if dmax > 1.0:
        data = data / dmax
    h, w = data.shape[:2]
    return DepthMap(data=data, resolution=(w, h), quality_score=0.8, edge_preservation=0.7)


def save_sample(output_dir: Path, index: int, rgb: np.ndarray, labels: dict) -> None:
    stem = f"synthetic_{index:04d}"
    np.save(output_dir / f"{stem}.npy", rgb)
    np.save(output_dir / f"{stem}_labels.npy", labels)


def generate_from_depth_files(
    generator: FLUXControlNetGenerator,
    depth_files: List[Path],
    output_dir: Path,
    weather_conditions: List[WeatherConditions],
    controlnet_scale: float,
) -> int:
    count = 0
    for i, depth_path in enumerate(tqdm(depth_files, desc="Synthetic from depth")):
        depth_map = load_depth_map(depth_path)
        conditions = weather_conditions[i % len(weather_conditions)]
        config = GenerationConfig(
            seed=42 + i,
            controlnet_conditioning_scale=controlnet_scale,
        )
        try:
            synthetic = generator.generate_wave_scene(depth_map, conditions, config)
            labels = {
                "height": synthetic.ground_truth_labels.height_meters,
                "height_meters": synthetic.ground_truth_labels.height_meters,
                "direction": synthetic.ground_truth_labels.direction,
                "breaking_type": synthetic.ground_truth_labels.breaking_type,
            }
            save_sample(output_dir, count, synthetic.rgb_data, labels)
            count += 1
        except Exception as exc:
            logger.warning("Failed %s: %s", depth_path.name, exc)
    return count


def main() -> int:
    args = parse_args()
    setup_logging()

    cfg = load_yaml_dict(args.config) if Path(args.config).exists() else {}
    syn = cfg.get("synthetic", {})
    num_images = args.num_images or syn.get("num_images", 100)
    controlnet_scale = float(syn.get("controlnet_conditioning_scale", 0.5))
    mode = args.mode or syn.get("mode", "from-depth")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        depth_files = list(Path(args.depth_dir).glob("*_depth.npy"))
        print(f"Would generate up to {num_images} images mode={mode} from {len(depth_files)} depth maps")
        return 0

    if not setup_hf_auth():
        logger.error("Set HF_TOKEN for FLUX model access")
        return 1

    logger.info("Initializing FLUX generator (this may take several minutes)...")
    generator = FLUXControlNetGenerator()
    weather = create_default_weather_conditions()

    if mode == "balanced":
        dataset = generator.create_balanced_dataset(target_size=num_images)
        for i, img in enumerate(dataset.images):
            labels = {
                "height": img.ground_truth_labels.height_meters,
                "direction": img.ground_truth_labels.direction,
                "breaking_type": img.ground_truth_labels.breaking_type,
            }
            save_sample(output_dir, i, img.rgb_data, labels)
        logger.info("Saved %s balanced synthetic images", len(dataset.images))
        return 0

    depth_files = sorted(Path(args.depth_dir).glob("*_depth.npy"))
    if args.limit:
        depth_files = depth_files[: args.limit]
    if not depth_files:
        logger.error("No *_depth.npy in %s — run extract_depth_maps.py first", args.depth_dir)
        return 1

    if num_images < len(depth_files):
        depth_files = depth_files[:num_images]

    saved = generate_from_depth_files(
        generator, depth_files, output_dir, weather, controlnet_scale
    )
    logger.info("Saved %s synthetic images to %s", saved, output_dir)
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
