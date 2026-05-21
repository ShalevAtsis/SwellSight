"""
PyTorch dataset classes for wave analysis training and evaluation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .manifest import DatasetManifest, load_manifest

logger = logging.getLogger(__name__)

DIR_MAP = {"LEFT": 0, "RIGHT": 1, "STRAIGHT": 2, "left": 0, "right": 1, "straight": 2}
BREAK_MAP = {
    "SPILLING": 0,
    "PLUNGING": 1,
    "SURGING": 2,
    "spilling": 0,
    "plunging": 1,
    "surging": 2,
}


@dataclass
class BeachCamImage:
    rgb_data: np.ndarray
    resolution: Tuple[int, int]
    format: str = "RGB"
    quality_score: float = 1.0


@dataclass
class DepthMap:
    data: np.ndarray


@dataclass
class WaveMetrics:
    height_meters: float
    direction: str
    breaking_type: str
    height_confidence: float = 1.0
    direction_confidence: float = 1.0
    breaking_confidence: float = 1.0


class DataSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DataSource(Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


@dataclass
class TrainingExample:
    image_id: str
    rgb_image: BeachCamImage
    depth_map: DepthMap
    labels: WaveMetrics
    data_source: DataSource
    augmentation_applied: List[str]


def _normalize_depth(depth_data: np.ndarray) -> np.ndarray:
    depth = depth_data.astype(np.float32)
    if depth.ndim == 3:
        depth = depth.squeeze()
    dmax = float(depth.max()) if depth.size else 1.0
    if dmax > 1.0:
        depth = depth / dmax
    return depth


def _load_depth_for_sample(
    stem: str, data_dir: Path, depth_dir: Optional[Path]
) -> Optional[np.ndarray]:
    candidates = [
        data_dir / f"{stem}_depth.npy",
    ]
    if depth_dir:
        candidates.append(depth_dir / f"{stem}_depth.npy")
    candidates.append(data_dir.parent / "depth_maps" / f"{stem}_depth.npy")

    for path in candidates:
        if path.exists():
            return _normalize_depth(np.load(path))

    return None


class WaveDataset(Dataset):
    """Loads SwellSight .npy training pairs (RGB + optional depth + labels)."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        transform: Optional[Any] = None,
        target_resolution: Tuple[int, int] = (518, 518),
        manifest_path: Optional[str] = None,
        depth_dir: Optional[str] = None,
        require_depth: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.require_depth = require_depth
        self.target_resolution = (
            (target_resolution[0] // 14) * 14,
            (target_resolution[1] // 14) * 14,
        )

        if manifest_path:
            self.examples = self._load_from_manifest(manifest_path)
        else:
            self.examples = self._load_and_split_data(train_ratio)

        logger.info(
            "[%s] %s examples from %s (resolution %s)",
            split.upper(),
            len(self.examples),
            self.data_dir,
            self.target_resolution,
        )

    def _parse_labels(self, raw_labels: Any) -> WaveMetrics:
        if isinstance(raw_labels, dict):
            return WaveMetrics(
                height_meters=float(raw_labels.get("height", raw_labels.get("height_meters", 0.0))),
                direction=str(raw_labels.get("direction", "STRAIGHT")).upper(),
                breaking_type=str(
                    raw_labels.get("breaking_type", raw_labels.get("breaking", "SPILLING"))
                ).upper(),
            )
        return raw_labels

    def _build_example(self, img_file: Path, stem: str) -> Optional[TrainingExample]:
        label_file = img_file.parent / f"{stem}_labels.npy"
        if not label_file.exists():
            return None

        rgb_data = np.load(img_file)
        raw_labels = np.load(label_file, allow_pickle=True).item()
        depth_data = _load_depth_for_sample(stem, img_file.parent, self.depth_dir)

        if depth_data is None:
            if self.require_depth:
                return None
            depth_data = np.zeros(rgb_data.shape[:2], dtype=np.float32)
        elif depth_data.shape[:2] != rgb_data.shape[:2]:
            depth_data = np.array(depth_data)
            depth_t = torch.from_numpy(depth_data).float().unsqueeze(0).unsqueeze(0)
            depth_data = (
                F.interpolate(
                    depth_t,
                    size=rgb_data.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )

        return TrainingExample(
            image_id=stem,
            rgb_image=BeachCamImage(rgb_data, rgb_data.shape[:2]),
            depth_map=DepthMap(depth_data),
            labels=self._parse_labels(raw_labels),
            data_source=DataSource.SYNTHETIC,
            augmentation_applied=[],
        )

    def _load_from_manifest(self, manifest_path: str) -> List[TrainingExample]:
        manifest: DatasetManifest = load_manifest(manifest_path)
        examples: List[TrainingExample] = []
        for entry in manifest.entries:
            if entry.split != self.split:
                continue
            img_file = self.data_dir / Path(entry.image_path).name
            if not img_file.exists():
                img_file = Path(entry.image_path)
            example = self._build_example(img_file, entry.id)
            if example:
                examples.append(example)
        return examples

    def _load_and_split_data(self, train_ratio: float) -> List[TrainingExample]:
        if not self.data_dir.exists():
            logger.warning("Data directory does not exist: %s", self.data_dir)
            return []

        all_files = sorted(
            f
            for f in self.data_dir.glob("*.npy")
            if "_labels" not in f.name and "_depth" not in f.name
        )
        random.Random(42).shuffle(all_files)
        split_idx = int(len(all_files) * train_ratio)
        selected = all_files[:split_idx] if self.split == "train" else all_files[split_idx:]

        examples: List[TrainingExample] = []
        for img_file in selected:
            try:
                example = self._build_example(img_file, img_file.stem)
                if example:
                    examples.append(example)
            except Exception as exc:
                logger.warning("Error loading %s: %s", img_file.name, exc)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        rgb_tensor = torch.from_numpy(example.rgb_image.rgb_data).float()
        if rgb_tensor.max() > 1.0:
            rgb_tensor /= 255.0

        depth_tensor = torch.from_numpy(example.depth_map.data).float()
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        if rgb_tensor.dim() == 3 and rgb_tensor.shape[-1] == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)

        if rgb_tensor.shape[1:] != self.target_resolution:
            rgb_tensor = (
                F.interpolate(
                    rgb_tensor.unsqueeze(0),
                    size=self.target_resolution,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            )

        if depth_tensor.shape[1:] != self.target_resolution:
            depth_tensor = (
                F.interpolate(
                    depth_tensor.unsqueeze(0),
                    size=self.target_resolution,
                    mode="nearest",
                ).squeeze(0)
            )

        input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)

        return {
            "input": input_tensor,
            "labels": {
                "height": torch.tensor(example.labels.height_meters, dtype=torch.float32),
                "direction": torch.tensor(
                    DIR_MAP.get(example.labels.direction.upper(), 2), dtype=torch.long
                ),
                "breaking_type": torch.tensor(
                    BREAK_MAP.get(example.labels.breaking_type.upper(), 0), dtype=torch.long
                ),
            },
        }
