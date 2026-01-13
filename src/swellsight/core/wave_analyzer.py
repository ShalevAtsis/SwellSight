"""
Stage C: Multi-Task Wave Analyzer

Unified model that simultaneously predicts wave height, direction, and breaking type
from RGB+Depth input using DINOv2 backbone.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import logging

from .depth_extractor import DepthMap
from .synthetic_generator import WaveMetrics
from ..models.backbone import DINOv2Backbone
from ..models.heads import WaveHeightHead, DirectionHead, BreakingTypeHead
from ..utils.hardware import HardwareManager
from ..utils.performance import PerformanceOptimizer, OptimizationConfig, PerformanceMetrics

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
    
    def __init__(self, backbone_model: str = "dinov2_vitb14", freeze_backbone: bool = True, 
                 device: Optional[str] = None, enable_optimization: bool = True):
        """Initialize DINOv2 wave analyzer.
        
        Args:
            backbone_model: DINOv2 model variant
            freeze_backbone: Whether to freeze backbone weights
            device: Device to run model on ("cuda", "cpu", or None for auto-detect)
            enable_optimization: Whether to enable performance optimizations
        """
        super().__init__()
        self.backbone_model = backbone_model
        self.freeze_backbone = freeze_backbone
        self.input_channels = 4  # RGB + Depth
        self.input_resolution = (518, 518)
        self._last_confidence = None
        self.enable_optimization = enable_optimization
        
        # Initialize hardware manager for optimal device selection
        self.hardware_manager = HardwareManager()
        
        # Auto-detect device if not specified using hardware manager
        if device is None:
            self.device = self.hardware_manager.get_device()
        else:
            self.device = torch.device(device)
        
        # Initialize performance optimizer
        if self.enable_optimization:
            optimization_config = OptimizationConfig(
                target_latency_ms=200.0,  # Target <200ms as per requirements
                enable_mixed_precision=True,
                enable_torch_compile=True,
                batch_size=1
            )
            self.performance_optimizer = PerformanceOptimizer(optimization_config)
        else:
            self.performance_optimizer = None
        
        # Initialize DINOv2 backbone
        self.backbone = DINOv2Backbone(backbone_model, freeze_backbone)
        
        # Initialize prediction heads
        feature_dim = self.backbone.get_feature_dim()
        self.height_head = WaveHeightHead(feature_dim)
        self.direction_head = DirectionHead(feature_dim)
        self.breaking_head = BreakingTypeHead(feature_dim)
        
        # Move model to device with memory optimization
        self._move_to_device_with_fallback()
        
        # Apply performance optimizations
        if self.enable_optimization:
            self._apply_optimizations()
        
        logger.info(f"Initialized DINOv2WaveAnalyzer with {backbone_model} on {self.device}")
        logger.info(f"Hardware info: {self.hardware_manager.hardware_info}")
        logger.info(f"Performance optimization: {'enabled' if enable_optimization else 'disabled'}")
    
    def _apply_optimizations(self):
        """Apply performance optimizations to the model."""
        if self.performance_optimizer is None:
            return
        
        logger.info("Applying performance optimizations...")
        
        # Optimize the entire model
        optimized_model = self.performance_optimizer.optimize_model(self)
        
        # Copy optimized parameters back (if torch.compile was applied)
        if hasattr(optimized_model, '_orig_mod'):
            # torch.compile was applied, keep reference to optimized version
            self._optimized_model = optimized_model
        else:
            self._optimized_model = self
        
        # Warmup the model for optimal performance
        input_shape = (self.input_channels, *self.input_resolution)
        self.performance_optimizer.warmup_model(self._optimized_model, input_shape)
        
        logger.info("Performance optimizations applied successfully")
    
    def _move_to_device_with_fallback(self):
        """Move model to device with automatic fallback on memory issues."""
        try:
            # Check memory requirements (approximate)
            model_memory_gb = 3.0  # Estimated memory for DINOv2 + heads
            
            if not self.hardware_manager.check_memory_requirements(model_memory_gb):
                logger.warning("Insufficient memory for GPU, falling back to CPU")
                self.device = torch.device("cpu")
            
            # Move components to device
            self.to(self.device)
            
            # Clean up GPU memory after loading
            if self.device.type == "cuda":
                self.hardware_manager.cleanup_gpu_memory()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU out of memory during model loading: {e}")
                logger.warning("Falling back to CPU")
                self.device = torch.device("cpu")
                self.to(self.device)
            else:
                raise e
        
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
        
    def analyze_waves(self, rgb_image: np.ndarray, depth_map: DepthMap) -> Tuple[WaveMetrics, Optional[PerformanceMetrics]]:
        """Analyze waves using DINOv2 multi-task model with performance optimization.
        
        Args:
            rgb_image: RGB beach cam image [H, W, 3]
            depth_map: Corresponding depth map
            
        Returns:
            Tuple of (WaveMetrics with all wave predictions, PerformanceMetrics if optimization enabled)
        """
        # Set model to evaluation mode
        self.eval()
        
        # Prepare 4-channel input
        input_tensor = self._prepare_input(rgb_image, depth_map)
        
        # Use optimized inference if available
        if self.enable_optimization and self.performance_optimizer:
            model_to_use = getattr(self, '_optimized_model', self)
            
            try:
                predictions, performance_metrics = self.performance_optimizer.optimize_inference(
                    model_to_use, input_tensor
                )
                
                # Convert to WaveMetrics
                wave_metrics = self._convert_to_wave_metrics(predictions)
                
                return wave_metrics, performance_metrics
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("GPU out of memory during optimized inference, falling back to standard inference")
                    self.hardware_manager.cleanup_gpu_memory()
                    # Fall through to standard inference
                else:
                    raise e
        
        # Standard inference (fallback or when optimization disabled)
        with torch.no_grad():
            # Move input to device
            input_tensor = input_tensor.to(self.device)
            
            # Get model predictions with memory management
            try:
                predictions = self.forward(input_tensor)
                
                # Convert to WaveMetrics
                wave_metrics = self._convert_to_wave_metrics(predictions)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("GPU out of memory during inference, cleaning up and retrying")
                    self.hardware_manager.cleanup_gpu_memory()
                    
                    # Retry with smaller batch or fallback to CPU
                    if self.device.type == "cuda":
                        logger.warning("Falling back to CPU for this inference")
                        input_tensor = input_tensor.cpu()
                        self.cpu()
                        predictions = self.forward(input_tensor)
                        wave_metrics = self._convert_to_wave_metrics(predictions)
                        self.to(self.device)  # Move back to original device
                    else:
                        raise e
                else:
                    raise e
            
            # Clean up GPU memory after inference
            if self.device.type == "cuda":
                self.hardware_manager.cleanup_gpu_memory()
        
        return wave_metrics, None
        
    def get_confidence_scores(self) -> ConfidenceScores:
        """Get confidence scores from last prediction."""
        if self._last_confidence is None:
            raise ValueError("No predictions made yet. Call analyze_waves() first.")
        return self._last_confidence
    
    def get_optimal_batch_size(self, input_height: int = 518, input_width: int = 518) -> int:
        """Get optimal batch size for current hardware configuration.
        
        Args:
            input_height: Input image height
            input_width: Input image width
            
        Returns:
            Optimal batch size for current hardware
        """
        # Estimate memory usage
        model_memory_mb = 3000  # Approximate model memory in MB
        input_size_mb = (input_height * input_width * 4 * 4) / (1024 * 1024)  # 4 channels, 4 bytes per float
        
        return self.hardware_manager.get_optimal_batch_size(model_memory_mb, input_size_mb)
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get comprehensive hardware information.
        
        Returns:
            Dictionary with hardware information
        """
        return self.hardware_manager.get_system_info()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from optimization history.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.enable_optimization or not self.performance_optimizer:
            return {"optimization_enabled": False}
        
        stats = self.performance_optimizer.get_performance_stats()
        stats["optimization_enabled"] = True
        stats["target_latency_ms"] = self.performance_optimizer.config.target_latency_ms
        
        return stats
    
    def clear_performance_history(self):
        """Clear performance monitoring history."""
        if self.enable_optimization and self.performance_optimizer:
            self.performance_optimizer.clear_performance_history()
    
    def is_real_time_capable(self) -> bool:
        """Check if the system is capable of real-time processing.
        
        Returns:
            True if average processing time is under target latency
        """
        if not self.enable_optimization or not self.performance_optimizer:
            return False
        
        stats = self.performance_optimizer.get_performance_stats()
        
        if not stats or "avg_total_time_ms" not in stats:
            return False
        
        return stats["avg_total_time_ms"] <= self.performance_optimizer.config.target_latency_ms