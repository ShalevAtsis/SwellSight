"""
Stage C: Multi-Task Wave Analyzer

Unified model that simultaneously predicts wave height, direction, and breaking type
from RGB+Depth input using DINOv2 backbone.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import logging

from .depth_extractor import DepthMap
from .synthetic_generator import WaveMetrics
from ..models.backbone import DINOv2Backbone
from ..models.heads import WaveHeightHead, DirectionHead, BreakingTypeHead

logger = logging.getLogger(__name__)

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

class DINOv2WaveAnalyzer(WaveAnalyzer, nn.Module):
    """DINOv2-based multi-task wave analyzer."""
    
    def __init__(self, backbone_model: str = "dinov2_vitb14", freeze_backbone: bool = True):
        """Initialize DINOv2 wave analyzer.
        
        Args:
            backbone_model: DINOv2 model variant
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        self.backbone_model = backbone_model
        self.freeze_backbone = freeze_backbone
        self.input_channels = 4  # RGB + Depth
        self.input_resolution = (518, 518)
        self._last_confidence = None
        
        # Initialize DINOv2 backbone
        self.backbone = DINOv2Backbone(backbone_model, freeze_backbone)
        
        # Initialize prediction heads
        feature_dim = self.backbone.get_feature_dim()
        self.height_head = WaveHeightHead(feature_dim)
        self.direction_head = DirectionHead(feature_dim)
        self.breaking_head = BreakingTypeHead(feature_dim)
        
        logger.info(f"Initialized DINOv2WaveAnalyzer with {backbone_model}")
        
    def _prepare_input(self, rgb_image: np.ndarray, depth_map: DepthMap) -> torch.Tensor:
        """Prepare 4-channel input tensor from RGB image and depth map.
        
        Args:
            rgb_image: RGB image as numpy array [H, W, 3]
            depth_map: Depth map object with normalized data
            
        Returns:
            4-channel tensor [1, 4, H, W] ready for model input
        """
        # Convert RGB to tensor and normalize
        if rgb_image.dtype == np.uint8:
            rgb_tensor = torch.from_numpy(rgb_image).float() / 255.0
        else:
            rgb_tensor = torch.from_numpy(rgb_image).float()
        
        # Convert depth map to tensor
        depth_tensor = torch.from_numpy(depth_map.data).float()
        
        # Ensure RGB is [H, W, 3] and depth is [H, W]
        if rgb_tensor.dim() == 3 and rgb_tensor.shape[-1] == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)  # [3, H, W]
        
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)  # [1, H, W]
        
        # Combine RGB and depth
        input_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)  # [4, H, W]
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)  # [1, 4, H, W]
        
        return input_tensor
    
    def _convert_to_wave_metrics(self, predictions: Dict[str, torch.Tensor]) -> WaveMetrics:
        """Convert model predictions to WaveMetrics object with enhanced height analysis.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            WaveMetrics object with all predictions
        """
        # Extract enhanced height predictions
        height_meters = predictions["height_meters"].item()
        height_feet = height_meters * 3.28084  # Convert to feet
        height_confidence = predictions["height_confidence"].item()
        
        # Extract additional height analysis data
        dominance_score = predictions.get("dominance_score", torch.tensor(1.0)).item()
        wave_count_estimate = predictions.get("wave_count_estimate", torch.tensor(1.0)).item()
        is_extreme_height = predictions.get("is_extreme_height", torch.tensor(0.0)).item() > 0.5
        
        # Extract enhanced direction predictions
        direction_probs = predictions["direction_probabilities"].squeeze()
        direction_idx = predictions["direction_predicted"].item()
        direction_confidence = predictions["direction_confidence"].item()
        direction_names = ["LEFT", "RIGHT", "STRAIGHT"]
        direction = direction_names[direction_idx]
        
        # Extract additional direction analysis
        mixed_conditions_score = predictions.get("mixed_conditions_score", torch.tensor(0.0)).item()
        direction_strength = predictions.get("direction_strength", torch.tensor(1.0)).item()
        wave_train_count = predictions.get("wave_train_count", torch.tensor(1.0)).item()
        is_mixed_conditions = predictions.get("is_mixed_conditions", torch.tensor(0.0)).item() > 0.5
        
        # Adjust direction confidence based on mixed conditions
        if is_mixed_conditions:
            direction_confidence *= 0.85  # Reduce confidence for mixed conditions
            
        # If multiple wave trains detected, note this affects confidence
        if wave_train_count > 2.0:
            direction_confidence *= 0.9  # Slight reduction for complex wave patterns
        
        # Extract enhanced breaking type predictions
        breaking_probs = predictions["breaking_probabilities"].squeeze()
        breaking_idx = predictions["breaking_predicted"].item()
        breaking_confidence = predictions["breaking_confidence"].item()
        breaking_names = ["SPILLING", "PLUNGING", "SURGING", "NO_BREAKING"]
        breaking_type = breaking_names[breaking_idx]
        
        # Extract additional breaking analysis
        breaking_intensity = predictions.get("breaking_intensity", torch.tensor(0.5)).item()
        mixed_breaking_score = predictions.get("mixed_breaking_score", torch.tensor(0.0)).item()
        breaking_clarity = predictions.get("breaking_clarity", torch.tensor(1.0)).item()
        no_breaking_score = predictions.get("no_breaking_score", torch.tensor(0.0)).item()
        is_mixed_breaking = predictions.get("is_mixed_breaking", torch.tensor(0.0)).item() > 0.5
        
        # Adjust breaking confidence based on analysis
        if is_mixed_breaking:
            breaking_confidence *= 0.85  # Reduce confidence for mixed breaking patterns
            
        if breaking_clarity < 0.5:
            breaking_confidence *= 0.9  # Reduce confidence for unclear patterns
            
        if breaking_type == "NO_BREAKING" and no_breaking_score > 0.7:
            breaking_confidence = max(breaking_confidence, 0.8)  # High confidence for clear no-breaking
        
        # Enhanced extreme conditions detection
        # Use both height-based detection and model's extreme detection
        height_extreme = height_meters < 0.5 or height_meters > 8.0
        extreme_conditions = height_extreme or is_extreme_height
        
        # Adjust confidence based on wave complexity and dominance
        # If multiple waves detected, reduce confidence slightly
        if wave_count_estimate > 2.0:
            height_confidence *= 0.95  # Slight reduction for complex wave scenarios
            
        # If dominance is low, this might not be the primary wave
        if dominance_score < 0.7:
            height_confidence *= 0.9  # Reduce confidence for non-dominant waves
        
        # Store enhanced confidence scores
        self._last_confidence = ConfidenceScores(
            height_confidence=height_confidence,
            direction_confidence=direction_confidence,
            breaking_type_confidence=breaking_confidence,
            overall_confidence=(height_confidence + direction_confidence + breaking_confidence) / 3.0
        )
        
        return WaveMetrics(
            height_meters=height_meters,
            height_feet=height_feet,
            height_confidence=height_confidence,
            direction=direction,
            direction_confidence=direction_confidence,
            breaking_type=breaking_type,
            breaking_confidence=breaking_confidence,
            extreme_conditions=extreme_conditions
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-task model.
        
        Args:
            x: 4-channel input tensor [B, 4, H, W]
            
        Returns:
            Dictionary of predictions from all heads
        """
        # Extract shared features using DINOv2 backbone
        features = self.backbone(x)
        
        # Get predictions from all heads
        height_pred = self.height_head(features)
        direction_pred = self.direction_head(features)
        breaking_pred = self.breaking_head(features)
        
        # Combine all predictions
        predictions = {
            **height_pred,
            **direction_pred,
            **breaking_pred
        }
        
        return predictions
        
    def analyze_waves(self, rgb_image: np.ndarray, depth_map: DepthMap) -> WaveMetrics:
        """Analyze waves using DINOv2 multi-task model.
        
        Args:
            rgb_image: RGB beach cam image [H, W, 3]
            depth_map: Corresponding depth map
            
        Returns:
            WaveMetrics with all wave predictions
        """
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Prepare 4-channel input
            input_tensor = self._prepare_input(rgb_image, depth_map)
            
            # Get model predictions
            predictions = self.forward(input_tensor)
            
            # Convert to WaveMetrics
            wave_metrics = self._convert_to_wave_metrics(predictions)
            
        return wave_metrics
        
    def get_confidence_scores(self) -> ConfidenceScores:
        """Get confidence scores from last prediction."""
        if self._last_confidence is None:
            raise ValueError("No predictions made yet. Call analyze_waves() first.")
        return self._last_confidence