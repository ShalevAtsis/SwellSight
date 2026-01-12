"""
File I/O utilities for SwellSight system.

Provides file management, data loading/saving, and path utilities
for the wave analysis system.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import pickle
import numpy as np
import torch
import cv2
from PIL import Image
import logging

class FileManager:
    """File I/O manager for SwellSight system."""
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """Initialize file manager.
        
        Args:
            base_dir: Base directory for file operations
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.logger = logging.getLogger(__name__)
    
    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Path object for the directory
        """
        dir_path = Path(path)
        if not dir_path.is_absolute():
            dir_path = self.base_dir / dir_path
        
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (RGB format)
        """
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = self.base_dir / image_path
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Use PIL for better format support
            image = Image.open(image_path)
            image_rgb = image.convert('RGB')
            return np.array(image_rgb)
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def save_image(self, 
                   image: np.ndarray, 
                   save_path: Union[str, Path],
                   format: str = "JPEG",
                   quality: int = 95) -> None:
        """Save image to file.
        
        Args:
            image: Image array (RGB format)
            save_path: Path to save image
            format: Image format ("JPEG", "PNG", "WEBP")
            quality: Image quality (for JPEG/WEBP)
        """
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.base_dir / save_path
        
        # Ensure directory exists
        self.ensure_dir(save_path.parent)
        
        try:
            image_pil = Image.fromarray(image.astype(np.uint8))
            
            if format.upper() in ["JPEG", "WEBP"]:
                image_pil.save(save_path, format=format, quality=quality)
            else:
                image_pil.save(save_path, format=format)
                
            self.logger.info(f"Image saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save image to {save_path}: {e}")
            raise
    
    def load_json(self, json_path: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON data from file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Loaded JSON data as dictionary
        """
        json_path = Path(json_path)
        if not json_path.is_absolute():
            json_path = self.base_dir / json_path
        
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {json_path}: {e}")
            raise
    
    def save_json(self, 
                  data: Dict[str, Any], 
                  save_path: Union[str, Path],
                  indent: int = 2) -> None:
        """Save data to JSON file.
        
        Args:
            data: Data to save
            save_path: Path to save JSON file
            indent: JSON indentation
        """
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.base_dir / save_path
        
        # Ensure directory exists
        self.ensure_dir(save_path.parent)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=indent, default=self._json_serializer)
            self.logger.info(f"JSON saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {save_path}: {e}")
            raise
    
    def load_model_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load PyTorch model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.base_dir / checkpoint_path
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
    
    def save_model_checkpoint(self, 
                            checkpoint: Dict[str, Any],
                            save_path: Union[str, Path]) -> None:
        """Save PyTorch model checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary to save
            save_path: Path to save checkpoint
        """
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.base_dir / save_path
        
        # Ensure directory exists
        self.ensure_dir(save_path.parent)
        
        try:
            torch.save(checkpoint, save_path)
            self.logger.info(f"Checkpoint saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {save_path}: {e}")
            raise
    
    def load_numpy_array(self, array_path: Union[str, Path]) -> np.ndarray:
        """Load numpy array from file.
        
        Args:
            array_path: Path to numpy array file (.npy or .npz)
            
        Returns:
            Loaded numpy array
        """
        array_path = Path(array_path)
        if not array_path.is_absolute():
            array_path = self.base_dir / array_path
        
        try:
            if array_path.suffix == '.npy':
                return np.load(array_path)
            elif array_path.suffix == '.npz':
                return np.load(array_path)
            else:
                raise ValueError(f"Unsupported numpy format: {array_path.suffix}")
        except Exception as e:
            self.logger.error(f"Failed to load numpy array from {array_path}: {e}")
            raise
    
    def save_numpy_array(self, 
                        array: np.ndarray,
                        save_path: Union[str, Path],
                        compressed: bool = False) -> None:
        """Save numpy array to file.
        
        Args:
            array: Numpy array to save
            save_path: Path to save array
            compressed: Whether to use compressed format
        """
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = self.base_dir / save_path
        
        # Ensure directory exists
        self.ensure_dir(save_path.parent)
        
        try:
            if compressed:
                np.savez_compressed(save_path, array)
            else:
                np.save(save_path, array)
            self.logger.info(f"Numpy array saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save numpy array to {save_path}: {e}")
            raise
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and torch objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")