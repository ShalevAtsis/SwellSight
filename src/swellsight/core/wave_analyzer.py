"""
Stage C: Multi-Task Wave Analyzer

Unified model that simultaneously predicts wave height, direction, and breaking type
from RGB+Depth input using DINOv2 backbone.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict
from dataclasses import dataclass
import numpy as np
from .depth_extractor import DepthMap
from .synthetic_generator import WaveMetrics

@dataclass
class ConfidenceScores:
    """Confidence scores for all wave predictions."""
    height_confidence: float
    direction_confidence: float
    breaking_type_confidence: float
    overall_confidence: float

class WaveAnalyzer(ABC):
    """Abstract base class for multi-task wave analysis."""
    
    @abstractmethod
    def analyze_waves(self, rgb_image: np.ndarray, depth_map: DepthMap) -> WaveMetrics:
        """Predict all wave metrics simultaneously.
        
        Args:
            rgb_image: RGB beach cam image
            depth_map: Corresponding depth map
            
        Returns:
            WaveMetrics with height, direction, and breaking type predictions
        """
        pass
    
    @abstractmethod
    def get_confidence_scores(self) -> ConfidenceScores:
        """Return confidence scores for all predictions.
        
        Returns:
            ConfidenceScores for the last prediction made
        """
        pass

class DINOv2WaveAnalyzer(WaveAnalyzer):
    """DINOv2-based multi-task wave analyzer."""
    
    def __init__(self, backbone_model: str = "dinov2-base", freeze_backbone: bool = True):
        """Initialize DINOv2 wave analyzer.
        
        Args:
            backbone_model: DINOv2 model variant
            freeze_backbone: Whether to freeze backbone weights
        """
        self.backbone_model = backbone_model
        self.freeze_backbone = freeze_backbone
        self.input_channels = 4  # RGB + Depth
        self.input_resolution = (518, 518)
        self._model = None
        self._last_confidence = None
        
    def analyze_waves(self, rgb_image: np.ndarray, depth_map: DepthMap) -> WaveMetrics:
        """Analyze waves using DINOv2 multi-task model."""
        # TODO: Implement DINOv2 multi-task analysis
        # This will be implemented in task 6.1-6.8
        raise NotImplementedError("Wave analysis will be implemented in task 6.1-6.8")
        
    def get_confidence_scores(self) -> ConfidenceScores:
        """Get confidence scores from last prediction."""
        if self._last_confidence is None:
            raise ValueError("No predictions made yet. Call analyze_waves() first.")
        return self._last_confidence