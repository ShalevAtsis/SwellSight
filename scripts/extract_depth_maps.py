#!/usr/bin/env python3
"""Extract depth maps from beach cam images using Depth-Anything-V2."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.core.depth_extractor import DepthAnythingV2Extractor, ProcessingError
from swellsight.utils.logging import setup_logging

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract depth maps for SwellSight training")
    parser.add_argument(
        "--input",
        "-i",
        default="data/raw/beach_cams",
        help="Directory of RGB images or single image file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/depth_maps",
        help="Directory for *_depth.npy and optional visualizations",
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "base", "large"],
        default="large",
        help="Depth-Anything-V2 model size",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp32",
        help="fp32 recommended on CPU / OpenCV-heavy paths",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Prefer GPU (default: auto via torch)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Skip depth maps below this quality score (0 = no filter)",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save grayscale visualization JPGs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images (for testing)",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def collect_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb(image_path: Path) -> Optional[np.ndarray]:
    bgr = cv2.imread(str(image_path))
    if bgr is None or bgr.size == 0:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_depth_maps(
    input_dir: str | Path,
    output_dir: str | Path,
    model_size: str = "large",
    precision: str = "fp32",
    use_gpu: bool = True,
    min_quality: float = 0.0,
    save_vis: bool = False,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """Extract depth maps; returns (success_count, total_count)."""
    import torch

    if use_gpu and not torch.cuda.is_available():
        logging.warning("GPU requested but CUDA not available; using CPU")

    extractor = DepthAnythingV2Extractor(
        model_size=model_size,
        precision=precision,
        enable_optimization=use_gpu and torch.cuda.is_available(),
    )

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    if limit is not None:
        images = images[:limit]

    if not images:
        logging.error("No images found under %s", input_path)
        return 0, 0

    logging.info("Processing %s images -> %s", len(images), output_path)
    success = 0

    for img_path in tqdm(images, desc="Depth extraction"):
        rgb = load_rgb(img_path)
        if rgb is None:
            logging.warning("Skip unreadable: %s", img_path)
            continue

        try:
            result = extractor.extract_depth(rgb)
        except ProcessingError as exc:
            logging.warning("Skip %s: %s", img_path.name, exc)
            continue

        depth_map = result[0] if isinstance(result, tuple) else result
        quality = getattr(depth_map, "quality_score", None)
        if quality is not None and quality < min_quality:
            logging.warning("Skip %s: quality %.3f < %.3f", img_path.name, quality, min_quality)
            continue

        stem = img_path.stem
        np.save(output_path / f"{stem}_depth.npy", depth_map.data)

        if save_vis:
            vis = cv2.normalize(depth_map.data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imwrite(str(output_path / f"{stem}_depth_vis.jpg"), vis)

        success += 1

    logging.info("Extracted %s / %s depth maps", success, len(images))
    return success, len(images)


def main() -> int:
    args = parse_args()
    setup_logging(level="DEBUG" if args.debug else "INFO")

    use_gpu = not args.no_gpu
    if args.gpu:
        use_gpu = True

    success, total = extract_depth_maps(
        input_dir=args.input,
        output_dir=args.output,
        model_size=args.model_size,
        precision=args.precision,
        use_gpu=use_gpu,
        min_quality=args.min_quality,
        save_vis=args.save_vis,
        limit=args.limit,
    )
    return 0 if success > 0 or total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
