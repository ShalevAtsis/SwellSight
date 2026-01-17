"""
PyTorch dataset classes for wave analysis training and evaluation.
Handles loading synthetic .npy files and creating training examples.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import random

# --- 1. DEFINITIONS FOR MISSING CLASSES ---
@dataclass
class BeachCamImage:
    """Container for RGB image data."""
    rgb_data: np.ndarray
    resolution: Tuple[int, int]
    format: str = "RGB"
    quality_score: float = 1.0

@dataclass
class DepthMap:
    """Container for Depth data."""
    data: np.ndarray

@dataclass
class WaveMetrics:
    """Container for Wave Labels."""
    height_meters: float
    direction: str
    breaking_type: str
    height_confidence: float = 1.0
    direction_confidence: float = 1.0
    breaking_confidence: float = 1.0

# --- 2. ENUMS & DATA CLASSES ---
class DataSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class DataSource(Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"

@dataclass
class TrainingExample:
    """Single training example with image, depth, and labels."""
    image_id: str
    rgb_image: BeachCamImage
    depth_map: DepthMap
    labels: WaveMetrics
    data_source: DataSource
    augmentation_applied: List[str]

# --- 3. DATASET IMPLEMENTATION ---
class WaveDataset(Dataset):
    """PyTorch dataset that loads SwellSight data from disk."""
    
    def __init__(self, 
                 data_dir: str, 
                 split: str = 'train', 
                 train_ratio: float = 0.8,
                 transform: Optional[Any] = None,
                 target_resolution: Tuple[int, int] = (224, 224)): # DINOv2 compatible (224 = 16*14)
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        # Ensure resolution is multiple of 14 for DINOv2
        self.target_resolution = (
            (target_resolution[0] // 14) * 14,
            (target_resolution[1] // 14) * 14
        )
        
        # Load and split files
        self.examples = self._load_and_split_data(train_ratio)
        
        print(f"[{split.upper()}] Loaded {len(self.examples)} examples from {self.data_dir}")
        print(f"[{split.upper()}] Target resolution: {self.target_resolution}")

    def _load_and_split_data(self, train_ratio: float) -> List[TrainingExample]:
        """Scans directory for .npy files and creates TrainingExamples."""
        if not self.data_dir.exists():
            print(f"Warning: Directory {self.data_dir} does not exist.")
            return []

        # Find all image files (excluding labels and depth files)
        # We look for files ending in .npy but not ending in _labels.npy or _depth.npy
        all_files = sorted([
            f for f in self.data_dir.glob('*.npy') 
            if '_labels' not in f.name and '_depth' not in f.name
        ])

        # Random Shuffle with fixed seed for consistent train/val splits
        random.Random(42).shuffle(all_files)
        
        # Split indices
        split_idx = int(len(all_files) * train_ratio)
        
        if self.split == 'train':
            selected_files = all_files[:split_idx]
        else: # validation
            selected_files = all_files[split_idx:]

        examples = []
        for img_file in selected_files:
            try:
                # 1. Load Image
                rgb_data = np.load(img_file)
                
                # 2. Load Label
                label_file = img_file.parent / f"{img_file.stem}_labels.npy"
                if not label_file.exists():
                    continue
                raw_labels = np.load(label_file, allow_pickle=True).item()
                
                # 3. Handle Missing Depth (Synthetic data might not have saved it in augmentation)
                # If 4-channel input is required, we need depth. 
                # For now, if missing, we generate a dummy zero depth map or skip.
                # Assuming synthetic data generation logic from previous steps:
                depth_data = np.zeros(rgb_data.shape[:2], dtype=np.float32) 
                
                # 4. Construct Objects
                image_obj = BeachCamImage(rgb_data, rgb_data.shape[:2])
                depth_obj = DepthMap(depth_data)
                
                # Handle dictionary vs object labels
                if isinstance(raw_labels, dict):
                    labels_obj = WaveMetrics(
                        height_meters=raw_labels.get('height', 0.0),
                        direction=raw_labels.get('direction', 'STRAIGHT'),
                        breaking_type=raw_labels.get('breaking_type', 'SPILLING')
                    )
                else:
                    labels_obj = raw_labels

                examples.append(TrainingExample(
                    image_id=img_file.stem,
                    rgb_image=image_obj,
                    depth_map=depth_obj,
                    labels=labels_obj,
                    data_source=DataSource.SYNTHETIC,
                    augmentation_applied=[]
                ))
            except Exception as e:
                print(f"Error loading {img_file.name}: {e}")
                continue
                
        return examples

    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(example.rgb_image.rgb_data).float()
        # Normalize to [0, 1] if currently [0, 255]
        if rgb_tensor.max() > 1.0:
            rgb_tensor /= 255.0

        depth_tensor = torch.from_numpy(example.depth_map.data).float()
        
        # Resize logic (Simple interpolation for now if sizes mismatch)
        # In a real pipeline, use transforms.Resize
        if rgb_tensor.shape[:2] != self.target_resolution:
            import torch.nn.functional as F
            # Permute for torch resize (C, H, W)
            rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
            rgb_tensor = F.interpolate(rgb_tensor, size=self.target_resolution, mode='bilinear', align_corners=False)
            rgb_tensor = rgb_tensor.squeeze(0) # Keep as (C, H, W)
        else:
             rgb_tensor = rgb_tensor.permute(2, 0, 1) # HWC -> CHW

        # Prepare 4-channel input
        if len(depth_tensor.shape) == 2:
            depth_tensor = depth_tensor.unsqueeze(0) # HW -> 1HW
            
        # Resize depth if needed
        if depth_tensor.shape[1:] != self.target_resolution:
             import torch.nn.functional as F
             depth_tensor = depth_tensor.unsqueeze(0)
             depth_tensor = F.interpolate(depth_tensor, size=self.target_resolution, mode='nearest')
             depth_tensor = depth_tensor.squeeze(0)

        # Concatenate RGB + Depth
        input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0) # 4, H, W
        
        # Map Strings to Indices
        dir_map = {"LEFT": 0, "RIGHT": 1, "STRAIGHT": 2}
        break_map = {"SPILLING": 0, "PLUNGING": 1, "SURGING": 2}
        
        return {
            "input": input_tensor,
            "labels": {
                "height": torch.tensor(example.labels.height_meters, dtype=torch.float32),
                "direction": torch.tensor(dir_map.get(example.labels.direction, 2), dtype=torch.long),
                "breaking_type": torch.tensor(break_map.get(example.labels.breaking_type, 0), dtype=torch.long)
            }
        }