"""
Quality validation and anomaly detection system for wave analysis.

This module provides comprehensive quality checks for input images, depth maps,
and predictions to ensure reliable wave analysis results.
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ImageQualityMetrics:
    """Quality metrics for input images."""
    resolution: Tuple[int, int]
    contrast_ratio: float
    brightness_mean: float
    sharpness_score: float
    ocean_coverage: float
    noise_level: float
    is_corrupted: bool
    quality_score: float  # Overall quality score [0, 1]
    
@dataclass
class DepthMapQualityMetrics:
    """Quality metrics for depth maps."""
    edge_preservation: float
    texture_capture: float
    far_field_sensitivity: float
    contrast_ratio: float
    depth_consistency: float
    statistical_validity: float
    quality_score: float  # Overall quality score [0, 1]
    
@dataclass
class PredictionAnomalyMetrics:
    """Anomaly detection metrics for predictions."""
    height_anomaly_score: float
    direction_anomaly_score: float
    breaking_anomaly_score: float
    confidence_anomaly_score: float
    overall_anomaly_score: float
    is_anomalous: bool
    anomaly_reasons: List[str]
    
@dataclass
class PerformanceDegradationMetrics:
    """Performance monitoring and degradation detection metrics."""
    processing_time_ms: float
    memory_usage_mb: float
    accuracy_trend: float
    confidence_trend: float
    error_rate: float
    is_degraded: bool
    degradation_reasons: List[str]

class QualityValidator(ABC):
    """Abstract base class for quality validation."""
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate data quality."""
        pass
    
    @abstractmethod
    def get_quality_metrics(self, data: Any) -> Any:
        """Get detailed quality metrics."""
        pass

class ImageQualityValidator(QualityValidator):
    """Validator for input image quality."""
    
    def __init__(self, min_resolution: Tuple[int, int] = (480, 480),
                 min_contrast: float = 0.1, min_ocean_coverage: float = 0.3):
        """Initialize image quality validator.
        
        Args:
            min_resolution: Minimum acceptable resolution (width, height)
            min_contrast: Minimum acceptable contrast ratio
            min_ocean_coverage: Minimum acceptable ocean coverage ratio
        """
        self.min_resolution = min_resolution
        self.min_contrast = min_contrast
        self.min_ocean_coverage = min_ocean_coverage
        
    def validate(self, image: np.ndarray) -> bool:
        """Validate image quality.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            True if image passes quality checks
        """
        try:
            metrics = self.get_quality_metrics(image)
            
            # Check basic quality criteria
            if metrics.is_corrupted:
                return False
            
            if metrics.resolution[0] < self.min_resolution[0] or metrics.resolution[1] < self.min_resolution[1]:
                return False
            
            if metrics.contrast_ratio < self.min_contrast:
                return False
            
            if metrics.ocean_coverage < self.min_ocean_coverage:
                return False
            
            # Overall quality score threshold
            if metrics.quality_score < 0.5:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image quality validation failed: {e}")
            return False
    
    def get_quality_metrics(self, image: np.ndarray) -> ImageQualityMetrics:
        """Get detailed image quality metrics.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            ImageQualityMetrics with detailed quality assessment
        """
        try:
            # Check for corruption
            is_corrupted = self._check_corruption(image)
            
            if is_corrupted:
                return ImageQualityMetrics(
                    resolution=(0, 0),
                    contrast_ratio=0.0,
                    brightness_mean=0.0,
                    sharpness_score=0.0,
                    ocean_coverage=0.0,
                    noise_level=1.0,
                    is_corrupted=True,
                    quality_score=0.0
                )
            
            # Get image dimensions
            if len(image.shape) == 3:
                height, width, channels = image.shape
            else:
                height, width = image.shape
                channels = 1
            
            # Convert to grayscale for analysis if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if channels == 3 else image[:, :, 0]
            else:
                gray = image
            
            # Compute quality metrics
            contrast_ratio = self._compute_contrast_ratio(gray)
            brightness_mean = np.mean(gray)
            sharpness_score = self._compute_sharpness(gray)
            ocean_coverage = self._estimate_ocean_coverage(image)
            noise_level = self._estimate_noise_level(gray)
            
            # Compute overall quality score
            quality_score = self._compute_overall_quality_score(
                contrast_ratio, sharpness_score, ocean_coverage, noise_level
            )
            
            return ImageQualityMetrics(
                resolution=(width, height),
                contrast_ratio=contrast_ratio,
                brightness_mean=brightness_mean / 255.0,  # Normalize to [0, 1]
                sharpness_score=sharpness_score,
                ocean_coverage=ocean_coverage,
                noise_level=noise_level,
                is_corrupted=False,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Failed to compute image quality metrics: {e}")
            return ImageQualityMetrics(
                resolution=(0, 0),
                contrast_ratio=0.0,
                brightness_mean=0.0,
                sharpness_score=0.0,
                ocean_coverage=0.0,
                noise_level=1.0,
                is_corrupted=True,
                quality_score=0.0
            )
    
    def _check_corruption(self, image: np.ndarray) -> bool:
        """Check if image is corrupted."""
        if image is None or image.size == 0:
            return True
        
        # Check for invalid values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            return True
        
        # Check for reasonable value ranges
        if image.dtype == np.uint8:
            if np.min(image) < 0 or np.max(image) > 255:
                return True
        elif image.dtype == np.float32 or image.dtype == np.float64:
            if np.min(image) < 0.0 or np.max(image) > 1.0:
                # Allow some tolerance for floating point images
                if np.min(image) < -0.1 or np.max(image) > 1.1:
                    return True
        
        return False
    
    def _compute_contrast_ratio(self, gray: np.ndarray) -> float:
        """Compute contrast ratio using standard deviation."""
        return float(np.std(gray) / 255.0)  # Normalize to [0, 1]
    
    def _compute_sharpness(self, gray: np.ndarray) -> float:
        """Compute sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        # Normalize to reasonable range [0, 1]
        return min(1.0, sharpness / 1000.0)
    
    def _estimate_ocean_coverage(self, image: np.ndarray) -> float:
        """Estimate ocean coverage in the image."""
        # Simple heuristic: assume ocean is in lower 2/3 of image and has blue-ish colors
        if len(image.shape) == 3:
            height, width, channels = image.shape
            ocean_region = image[height//3:, :, :]  # Lower 2/3 of image
            
            # Look for blue-ish pixels (simple heuristic)
            if channels >= 3:
                blue_mask = (ocean_region[:, :, 2] > ocean_region[:, :, 0]) & \
                           (ocean_region[:, :, 2] > ocean_region[:, :, 1])
                ocean_coverage = np.sum(blue_mask) / (ocean_region.shape[0] * ocean_region.shape[1])
            else:
                # For grayscale, assume darker regions are water
                ocean_coverage = np.sum(ocean_region < 128) / (ocean_region.shape[0] * ocean_region.shape[1])
        else:
            # For grayscale images
            height, width = image.shape
            ocean_region = image[height//3:, :]
            ocean_coverage = np.sum(ocean_region < 128) / (ocean_region.shape[0] * ocean_region.shape[1])
        
        return float(ocean_coverage)
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level in the image."""
        # Use high-pass filter to estimate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        noise_level = np.std(filtered) / 255.0
        return min(1.0, noise_level)
    
    def _compute_overall_quality_score(self, contrast: float, sharpness: float, 
                                     ocean_coverage: float, noise_level: float) -> float:
        """Compute overall quality score from individual metrics."""
        # Weighted combination of quality factors
        quality = (
            0.3 * contrast +           # 30% weight on contrast
            0.3 * sharpness +          # 30% weight on sharpness
            0.3 * ocean_coverage +     # 30% weight on ocean coverage
            0.1 * (1.0 - noise_level)  # 10% weight on low noise
        )
        return min(1.0, max(0.0, quality))

class DepthMapQualityValidator(QualityValidator):
    """Validator for depth map quality."""
    
    def __init__(self, min_edge_preservation: float = 0.3,
                 min_contrast: float = 0.1):
        """Initialize depth map quality validator.
        
        Args:
            min_edge_preservation: Minimum acceptable edge preservation score
            min_contrast: Minimum acceptable contrast ratio
        """
        self.min_edge_preservation = min_edge_preservation
        self.min_contrast = min_contrast
    
    def validate(self, depth_map: np.ndarray) -> bool:
        """Validate depth map quality.
        
        Args:
            depth_map: Depth map as numpy array
            
        Returns:
            True if depth map passes quality checks
        """
        try:
            metrics = self.get_quality_metrics(depth_map)
            
            if metrics.edge_preservation < self.min_edge_preservation:
                return False
            
            if metrics.contrast_ratio < self.min_contrast:
                return False
            
            if metrics.quality_score < 0.5:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Depth map quality validation failed: {e}")
            return False
    
    def get_quality_metrics(self, depth_map: np.ndarray) -> DepthMapQualityMetrics:
        """Get detailed depth map quality metrics.
        
        Args:
            depth_map: Depth map as numpy array
            
        Returns:
            DepthMapQualityMetrics with detailed quality assessment
        """
        try:
            # Compute quality metrics
            edge_preservation = self._compute_edge_preservation(depth_map)
            texture_capture = self._compute_texture_capture(depth_map)
            far_field_sensitivity = self._compute_far_field_sensitivity(depth_map)
            contrast_ratio = self._compute_contrast_ratio(depth_map)
            depth_consistency = self._compute_depth_consistency(depth_map)
            statistical_validity = self._compute_statistical_validity(depth_map)
            
            # Compute overall quality score
            quality_score = self._compute_overall_quality_score(
                edge_preservation, texture_capture, far_field_sensitivity,
                contrast_ratio, depth_consistency, statistical_validity
            )
            
            return DepthMapQualityMetrics(
                edge_preservation=edge_preservation,
                texture_capture=texture_capture,
                far_field_sensitivity=far_field_sensitivity,
                contrast_ratio=contrast_ratio,
                depth_consistency=depth_consistency,
                statistical_validity=statistical_validity,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Failed to compute depth map quality metrics: {e}")
            return DepthMapQualityMetrics(
                edge_preservation=0.0,
                texture_capture=0.0,
                far_field_sensitivity=0.0,
                contrast_ratio=0.0,
                depth_consistency=0.0,
                statistical_validity=0.0,
                quality_score=0.0
            )
    
    def _compute_edge_preservation(self, depth_map: np.ndarray) -> float:
        """Compute edge preservation score."""
        # Use Sobel edge detection
        sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize edge strength
        edge_score = np.mean(edge_magnitude)
        return min(1.0, edge_score)
    
    def _compute_texture_capture(self, depth_map: np.ndarray) -> float:
        """Compute texture capture score."""
        # Use local standard deviation as texture measure
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(depth_map, -1, kernel)
        local_variance = cv2.filter2D(depth_map**2, -1, kernel) - local_mean**2
        texture_score = np.mean(np.sqrt(local_variance))
        return min(1.0, texture_score)
    
    def _compute_far_field_sensitivity(self, depth_map: np.ndarray) -> float:
        """Compute far field sensitivity score."""
        # Check depth variation in far regions (upper part of image)
        height = depth_map.shape[0]
        far_region = depth_map[:height//3, :]  # Upper 1/3 of image
        
        # Measure depth variation in far field
        far_field_std = np.std(far_region)
        sensitivity_score = min(1.0, far_field_std * 2.0)  # Scale appropriately
        return sensitivity_score
    
    def _compute_contrast_ratio(self, depth_map: np.ndarray) -> float:
        """Compute contrast ratio."""
        return float(np.std(depth_map))
    
    def _compute_depth_consistency(self, depth_map: np.ndarray) -> float:
        """Compute depth consistency score."""
        # Check for smooth depth transitions (avoid sudden jumps)
        grad_x = np.gradient(depth_map, axis=1)
        grad_y = np.gradient(depth_map, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Penalize very large gradients (inconsistent depth)
        large_gradients = gradient_magnitude > np.percentile(gradient_magnitude, 95)
        consistency_score = 1.0 - np.mean(large_gradients)
        return consistency_score
    
    def _compute_statistical_validity(self, depth_map: np.ndarray) -> float:
        """Compute statistical validity score."""
        # Check for reasonable depth distribution
        depth_mean = np.mean(depth_map)
        depth_std = np.std(depth_map)
        
        # Check for reasonable range [0, 1] for normalized depth
        if depth_mean < 0.0 or depth_mean > 1.0:
            return 0.0
        
        # Check for reasonable variation
        if depth_std < 0.01:  # Too uniform
            return 0.5
        
        if depth_std > 0.5:   # Too variable
            return 0.5
        
        return 1.0
    
    def _compute_overall_quality_score(self, edge_preservation: float, texture_capture: float,
                                     far_field_sensitivity: float, contrast_ratio: float,
                                     depth_consistency: float, statistical_validity: float) -> float:
        """Compute overall quality score from individual metrics."""
        quality = (
            0.25 * edge_preservation +      # 25% weight on edge preservation
            0.20 * texture_capture +        # 20% weight on texture capture
            0.15 * far_field_sensitivity +  # 15% weight on far field sensitivity
            0.15 * contrast_ratio +         # 15% weight on contrast
            0.15 * depth_consistency +      # 15% weight on consistency
            0.10 * statistical_validity     # 10% weight on statistical validity
        )
        return min(1.0, max(0.0, quality))

class PredictionAnomalyDetector:
    """Detector for anomalous predictions."""
    
    def __init__(self, contamination: float = 0.1):
        """Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in the data
        """
        self.contamination = contamination
        self.height_detector = IsolationForest(contamination=contamination, random_state=42)
        self.direction_detector = IsolationForest(contamination=contamination, random_state=42)
        self.breaking_detector = IsolationForest(contamination=contamination, random_state=42)
        self.confidence_detector = IsolationForest(contamination=contamination, random_state=42)
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Historical data for training
        self.prediction_history = []
        
    def fit(self, prediction_history: List[Dict[str, float]]) -> None:
        """Fit anomaly detectors on historical prediction data.
        
        Args:
            prediction_history: List of prediction dictionaries with metrics
        """
        if len(prediction_history) < 10:
            logger.warning("Insufficient data for anomaly detection training")
            return
        
        try:
            # Extract features for each task
            height_features = []
            direction_features = []
            breaking_features = []
            confidence_features = []
            
            for pred in prediction_history:
                # Height features: value, confidence, extreme flag
                height_features.append([
                    pred.get('height_meters', 0.0),
                    pred.get('height_confidence', 0.0),
                    float(pred.get('extreme_conditions', False))
                ])
                
                # Direction features: confidence, mixed conditions
                direction_features.append([
                    pred.get('direction_confidence', 0.0),
                    pred.get('mixed_conditions_score', 0.0)
                ])
                
                # Breaking features: confidence, intensity, clarity
                breaking_features.append([
                    pred.get('breaking_confidence', 0.0),
                    pred.get('breaking_intensity', 0.0),
                    pred.get('breaking_clarity', 1.0)
                ])
                
                # Overall confidence features
                confidence_features.append([
                    pred.get('height_confidence', 0.0),
                    pred.get('direction_confidence', 0.0),
                    pred.get('breaking_confidence', 0.0),
                    pred.get('overall_confidence', 0.0)
                ])
            
            # Convert to numpy arrays
            height_features = np.array(height_features)
            direction_features = np.array(direction_features)
            breaking_features = np.array(breaking_features)
            confidence_features = np.array(confidence_features)
            
            # Fit detectors
            self.height_detector.fit(height_features)
            self.direction_detector.fit(direction_features)
            self.breaking_detector.fit(breaking_features)
            self.confidence_detector.fit(confidence_features)
            
            self.is_fitted = True
            logger.info(f"Anomaly detectors fitted on {len(prediction_history)} samples")
            
        except Exception as e:
            logger.error(f"Failed to fit anomaly detectors: {e}")
            self.is_fitted = False
    
    def detect_anomalies(self, prediction: Dict[str, Any]) -> PredictionAnomalyMetrics:
        """Detect anomalies in a prediction.
        
        Args:
            prediction: Prediction dictionary with wave metrics
            
        Returns:
            PredictionAnomalyMetrics with anomaly analysis
        """
        if not self.is_fitted:
            # Return default metrics if not fitted
            return PredictionAnomalyMetrics(
                height_anomaly_score=0.0,
                direction_anomaly_score=0.0,
                breaking_anomaly_score=0.0,
                confidence_anomaly_score=0.0,
                overall_anomaly_score=0.0,
                is_anomalous=False,
                anomaly_reasons=["Anomaly detector not fitted"]
            )
        
        try:
            anomaly_reasons = []
            
            # Extract features for each task
            height_features = np.array([[
                prediction.get('height_meters', 0.0),
                prediction.get('height_confidence', 0.0),
                float(prediction.get('extreme_conditions', False))
            ]])
            
            direction_features = np.array([[
                prediction.get('direction_confidence', 0.0),
                prediction.get('mixed_conditions_score', 0.0)
            ]])
            
            breaking_features = np.array([[
                prediction.get('breaking_confidence', 0.0),
                prediction.get('breaking_intensity', 0.0),
                prediction.get('breaking_clarity', 1.0)
            ]])
            
            confidence_features = np.array([[
                prediction.get('height_confidence', 0.0),
                prediction.get('direction_confidence', 0.0),
                prediction.get('breaking_confidence', 0.0),
                prediction.get('overall_confidence', 0.0)
            ]])
            
            # Get anomaly scores (negative values indicate anomalies)
            height_score = self.height_detector.decision_function(height_features)[0]
            direction_score = self.direction_detector.decision_function(direction_features)[0]
            breaking_score = self.breaking_detector.decision_function(breaking_features)[0]
            confidence_score = self.confidence_detector.decision_function(confidence_features)[0]
            
            # Convert to [0, 1] range (higher = more anomalous)
            height_anomaly = max(0.0, -height_score)
            direction_anomaly = max(0.0, -direction_score)
            breaking_anomaly = max(0.0, -breaking_score)
            confidence_anomaly = max(0.0, -confidence_score)
            
            # Overall anomaly score
            overall_anomaly = (height_anomaly + direction_anomaly + breaking_anomaly + confidence_anomaly) / 4.0
            
            # Check for specific anomaly conditions
            if height_anomaly > 0.5:
                anomaly_reasons.append("Unusual wave height pattern")
            
            if direction_anomaly > 0.5:
                anomaly_reasons.append("Unusual direction classification pattern")
            
            if breaking_anomaly > 0.5:
                anomaly_reasons.append("Unusual breaking type pattern")
            
            if confidence_anomaly > 0.5:
                anomaly_reasons.append("Unusual confidence score pattern")
            
            # Additional rule-based anomaly checks
            if prediction.get('height_meters', 0.0) > 10.0:
                anomaly_reasons.append("Extremely high wave height (>10m)")
                overall_anomaly = max(overall_anomaly, 0.8)
            
            if prediction.get('height_confidence', 1.0) < 0.1:
                anomaly_reasons.append("Extremely low height confidence")
                overall_anomaly = max(overall_anomaly, 0.6)
            
            if all(conf < 0.2 for conf in [
                prediction.get('height_confidence', 1.0),
                prediction.get('direction_confidence', 1.0),
                prediction.get('breaking_confidence', 1.0)
            ]):
                anomaly_reasons.append("All confidence scores extremely low")
                overall_anomaly = max(overall_anomaly, 0.7)
            
            is_anomalous = overall_anomaly > 0.5 or len(anomaly_reasons) > 0
            
            return PredictionAnomalyMetrics(
                height_anomaly_score=height_anomaly,
                direction_anomaly_score=direction_anomaly,
                breaking_anomaly_score=breaking_anomaly,
                confidence_anomaly_score=confidence_anomaly,
                overall_anomaly_score=overall_anomaly,
                is_anomalous=is_anomalous,
                anomaly_reasons=anomaly_reasons
            )
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return PredictionAnomalyMetrics(
                height_anomaly_score=0.0,
                direction_anomaly_score=0.0,
                breaking_anomaly_score=0.0,
                confidence_anomaly_score=0.0,
                overall_anomaly_score=0.0,
                is_anomalous=False,
                anomaly_reasons=[f"Anomaly detection failed: {e}"]
            )

class PerformanceMonitor:
    """Monitor system performance and detect degradation."""
    
    def __init__(self, history_size: int = 100):
        """Initialize performance monitor.
        
        Args:
            history_size: Number of recent measurements to keep
        """
        self.history_size = history_size
        self.processing_times = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        self.accuracy_scores = deque(maxlen=history_size)
        self.confidence_scores = deque(maxlen=history_size)
        self.error_counts = deque(maxlen=history_size)
        
        # Baseline performance metrics
        self.baseline_processing_time = None
        self.baseline_accuracy = None
        self.baseline_confidence = None
        
    def record_performance(self, processing_time_ms: float, memory_usage_mb: float,
                         accuracy_score: Optional[float] = None,
                         confidence_score: Optional[float] = None,
                         had_error: bool = False) -> None:
        """Record performance metrics.
        
        Args:
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory usage in megabytes
            accuracy_score: Accuracy score if available
            confidence_score: Confidence score if available
            had_error: Whether an error occurred
        """
        self.processing_times.append(processing_time_ms)
        self.memory_usage.append(memory_usage_mb)
        
        if accuracy_score is not None:
            self.accuracy_scores.append(accuracy_score)
        
        if confidence_score is not None:
            self.confidence_scores.append(confidence_score)
        
        self.error_counts.append(1 if had_error else 0)
        
        # Update baselines if this is early in the monitoring
        if len(self.processing_times) == 10:  # After 10 samples, set baseline
            self.baseline_processing_time = np.mean(list(self.processing_times))
            if self.accuracy_scores:
                self.baseline_accuracy = np.mean(list(self.accuracy_scores))
            if self.confidence_scores:
                self.baseline_confidence = np.mean(list(self.confidence_scores))
    
    def detect_degradation(self) -> PerformanceDegradationMetrics:
        """Detect performance degradation.
        
        Returns:
            PerformanceDegradationMetrics with degradation analysis
        """
        if len(self.processing_times) < 10:
            return PerformanceDegradationMetrics(
                processing_time_ms=0.0,
                memory_usage_mb=0.0,
                accuracy_trend=0.0,
                confidence_trend=0.0,
                error_rate=0.0,
                is_degraded=False,
                degradation_reasons=["Insufficient data for degradation detection"]
            )
        
        try:
            degradation_reasons = []
            
            # Current metrics
            current_processing_time = np.mean(list(self.processing_times)[-10:])  # Last 10 samples
            current_memory_usage = np.mean(list(self.memory_usage)[-10:])
            current_error_rate = np.mean(list(self.error_counts)[-10:])
            
            # Trends
            processing_times_array = np.array(list(self.processing_times))
            accuracy_trend = self._compute_trend(list(self.accuracy_scores)) if self.accuracy_scores else 0.0
            confidence_trend = self._compute_trend(list(self.confidence_scores)) if self.confidence_scores else 0.0
            
            # Check for degradation conditions
            is_degraded = False
            
            # Processing time degradation
            if self.baseline_processing_time and current_processing_time > self.baseline_processing_time * 1.5:
                degradation_reasons.append(f"Processing time increased by {((current_processing_time / self.baseline_processing_time - 1) * 100):.1f}%")
                is_degraded = True
            
            # Memory usage degradation
            if current_memory_usage > 1000:  # More than 1GB
                degradation_reasons.append(f"High memory usage: {current_memory_usage:.1f}MB")
                is_degraded = True
            
            # Accuracy degradation
            if self.baseline_accuracy and accuracy_trend < -0.1:  # 10% drop in accuracy trend
                degradation_reasons.append(f"Accuracy trend declining: {accuracy_trend:.3f}")
                is_degraded = True
            
            # Confidence degradation
            if self.baseline_confidence and confidence_trend < -0.1:  # 10% drop in confidence trend
                degradation_reasons.append(f"Confidence trend declining: {confidence_trend:.3f}")
                is_degraded = True
            
            # Error rate degradation
            if current_error_rate > 0.1:  # More than 10% error rate
                degradation_reasons.append(f"High error rate: {current_error_rate * 100:.1f}%")
                is_degraded = True
            
            return PerformanceDegradationMetrics(
                processing_time_ms=current_processing_time,
                memory_usage_mb=current_memory_usage,
                accuracy_trend=accuracy_trend,
                confidence_trend=confidence_trend,
                error_rate=current_error_rate,
                is_degraded=is_degraded,
                degradation_reasons=degradation_reasons
            )
            
        except Exception as e:
            logger.error(f"Failed to detect performance degradation: {e}")
            return PerformanceDegradationMetrics(
                processing_time_ms=0.0,
                memory_usage_mb=0.0,
                accuracy_trend=0.0,
                confidence_trend=0.0,
                error_rate=0.0,
                is_degraded=False,
                degradation_reasons=[f"Degradation detection failed: {e}"]
            )
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend in values using linear regression slope."""
        if len(values) < 5:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary.
        
        Returns:
            Dictionary with performance summary statistics
        """
        if not self.processing_times:
            return {"status": "No data available"}
        
        summary = {
            "samples_collected": len(self.processing_times),
            "avg_processing_time_ms": np.mean(list(self.processing_times)),
            "avg_memory_usage_mb": np.mean(list(self.memory_usage)),
            "error_rate": np.mean(list(self.error_counts)),
            "baseline_processing_time_ms": self.baseline_processing_time,
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_confidence": self.baseline_confidence
        }
        
        if self.accuracy_scores:
            summary["avg_accuracy"] = np.mean(list(self.accuracy_scores))
            summary["accuracy_trend"] = self._compute_trend(list(self.accuracy_scores))
        
        if self.confidence_scores:
            summary["avg_confidence"] = np.mean(list(self.confidence_scores))
            summary["confidence_trend"] = self._compute_trend(list(self.confidence_scores))
        
        return summary

class ComprehensiveQualityValidator:
    """Comprehensive quality validation system combining all validators."""
    
    def __init__(self):
        """Initialize comprehensive quality validator."""
        self.image_validator = ImageQualityValidator()
        self.depth_validator = DepthMapQualityValidator()
        self.anomaly_detector = PredictionAnomalyDetector()
        self.performance_monitor = PerformanceMonitor()
        
        logger.info("Initialized ComprehensiveQualityValidator")
    
    def validate_input_quality(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Validate input image and depth map quality.
        
        Args:
            rgb_image: RGB input image
            depth_map: Depth map array
            
        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        try:
            # Validate image quality
            image_valid = self.image_validator.validate(rgb_image)
            image_metrics = self.image_validator.get_quality_metrics(rgb_image)
            
            # Validate depth map quality
            depth_valid = self.depth_validator.validate(depth_map)
            depth_metrics = self.depth_validator.get_quality_metrics(depth_map)
            
            # Overall validation
            is_valid = image_valid and depth_valid
            
            quality_metrics = {
                "image_valid": image_valid,
                "depth_valid": depth_valid,
                "overall_valid": is_valid,
                "image_metrics": image_metrics,
                "depth_metrics": depth_metrics
            }
            
            return is_valid, quality_metrics
            
        except Exception as e:
            logger.error(f"Input quality validation failed: {e}")
            return False, {"error": str(e)}
    
    def validate_prediction_quality(self, prediction: Dict[str, Any]) -> Tuple[bool, PredictionAnomalyMetrics]:
        """Validate prediction quality and detect anomalies.
        
        Args:
            prediction: Prediction dictionary with wave metrics
            
        Returns:
            Tuple of (is_valid, anomaly_metrics)
        """
        try:
            anomaly_metrics = self.anomaly_detector.detect_anomalies(prediction)
            is_valid = not anomaly_metrics.is_anomalous
            
            return is_valid, anomaly_metrics
            
        except Exception as e:
            logger.error(f"Prediction quality validation failed: {e}")
            return False, PredictionAnomalyMetrics(
                height_anomaly_score=0.0,
                direction_anomaly_score=0.0,
                breaking_anomaly_score=0.0,
                confidence_anomaly_score=0.0,
                overall_anomaly_score=0.0,
                is_anomalous=False,
                anomaly_reasons=[f"Validation failed: {e}"]
            )
    
    def monitor_performance(self, processing_time_ms: float, memory_usage_mb: float,
                          accuracy_score: Optional[float] = None,
                          confidence_score: Optional[float] = None,
                          had_error: bool = False) -> PerformanceDegradationMetrics:
        """Monitor system performance and detect degradation.
        
        Args:
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory usage in megabytes
            accuracy_score: Accuracy score if available
            confidence_score: Confidence score if available
            had_error: Whether an error occurred
            
        Returns:
            PerformanceDegradationMetrics with degradation analysis
        """
        self.performance_monitor.record_performance(
            processing_time_ms, memory_usage_mb, accuracy_score, confidence_score, had_error
        )
        
        return self.performance_monitor.detect_degradation()
    
    def fit_anomaly_detector(self, prediction_history: List[Dict[str, float]]) -> None:
        """Fit anomaly detector on historical prediction data.
        
        Args:
            prediction_history: List of prediction dictionaries
        """
        self.anomaly_detector.fit(prediction_history)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report.
        
        Returns:
            Dictionary with system health metrics
        """
        try:
            performance_summary = self.performance_monitor.get_performance_summary()
            degradation_metrics = self.performance_monitor.detect_degradation()
            
            health_report = {
                "timestamp": time.time(),
                "performance_summary": performance_summary,
                "degradation_status": {
                    "is_degraded": degradation_metrics.is_degraded,
                    "degradation_reasons": degradation_metrics.degradation_reasons,
                    "processing_time_ms": degradation_metrics.processing_time_ms,
                    "memory_usage_mb": degradation_metrics.memory_usage_mb,
                    "error_rate": degradation_metrics.error_rate
                },
                "anomaly_detector_status": {
                    "is_fitted": self.anomaly_detector.is_fitted,
                    "contamination": self.anomaly_detector.contamination
                }
            }
            
            return health_report
            
        except Exception as e:
            logger.error(f"Failed to generate system health report: {e}")
            return {"error": str(e), "timestamp": time.time()}