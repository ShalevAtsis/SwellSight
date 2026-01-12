"""
Data Validation Utilities for SwellSight Pipeline
Handles image quality validation, depth map assessment, and data integrity checks
"""

import os
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """Handles data validation and quality assessment for pipeline stages"""
    
    # Image quality thresholds
    MIN_RESOLUTION = (224, 224)  # Minimum acceptable resolution
    MAX_RESOLUTION = (4096, 4096)  # Maximum reasonable resolution
    MIN_FILE_SIZE = 1024  # Minimum file size in bytes (1KB)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # Maximum file size in bytes (50MB)
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
    
    def validate_image_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Validate image quality with comprehensive checks
        Returns quality metrics and validation status
        """
        try:
            image_path = Path(image_path)
            
            # Check file existence
            if not image_path.exists():
                return {
                    'valid': False,
                    'score': 0.0,
                    'issues': ['File does not exist'],
                    'metrics': {}
                }
            
            # Check file size
            file_size = image_path.stat().st_size
            if file_size < self.MIN_FILE_SIZE:
                return {
                    'valid': False,
                    'score': 0.0,
                    'issues': [f'File too small: {file_size} bytes'],
                    'metrics': {'file_size': file_size}
                }
            
            if file_size > self.MAX_FILE_SIZE:
                return {
                    'valid': False,
                    'score': 0.0,
                    'issues': [f'File too large: {file_size} bytes'],
                    'metrics': {'file_size': file_size}
                }
            
            # Check file extension
            if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return {
                    'valid': False,
                    'score': 0.0,
                    'issues': [f'Unsupported format: {image_path.suffix}'],
                    'metrics': {'file_size': file_size}
                }
            
            # Try to load and validate image
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode
                    
                    # Check resolution
                    if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                        return {
                            'valid': False,
                            'score': 0.0,
                            'issues': [f'Resolution too low: {width}x{height}'],
                            'metrics': {
                                'width': width,
                                'height': height,
                                'mode': mode,
                                'file_size': file_size
                            }
                        }
                    
                    if width > self.MAX_RESOLUTION[0] or height > self.MAX_RESOLUTION[1]:
                        return {
                            'valid': False,
                            'score': 0.0,
                            'issues': [f'Resolution too high: {width}x{height}'],
                            'metrics': {
                                'width': width,
                                'height': height,
                                'mode': mode,
                                'file_size': file_size
                            }
                        }
                    
                    # Convert to numpy array for quality analysis
                    img_array = np.array(img)
                    
                    # Calculate quality metrics
                    quality_metrics = self._calculate_image_quality_metrics(img_array)
                    
                    # Determine overall quality score
                    quality_score = self._calculate_quality_score(quality_metrics)
                    
                    # Check for issues
                    issues = []
                    if quality_metrics['brightness_std'] < 10:
                        issues.append('Very low brightness variation (possibly corrupted)')
                    if quality_metrics['contrast'] < 0.1:
                        issues.append('Very low contrast')
                    if quality_metrics['sharpness'] < 0.1:
                        issues.append('Very low sharpness (possibly blurred)')
                    
                    return {
                        'valid': quality_score >= self.quality_threshold and len(issues) == 0,
                        'score': quality_score,
                        'issues': issues,
                        'metrics': {
                            'width': width,
                            'height': height,
                            'mode': mode,
                            'file_size': file_size,
                            **quality_metrics
                        }
                    }
                    
            except Exception as e:
                return {
                    'valid': False,
                    'score': 0.0,
                    'issues': [f'Image loading error: {str(e)}'],
                    'metrics': {'file_size': file_size}
                }
                
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {e}")
            return {
                'valid': False,
                'score': 0.0,
                'issues': [f'Validation error: {str(e)}'],
                'metrics': {}
            }
    
    def _calculate_image_quality_metrics(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculate various image quality metrics"""
        try:
            # Convert to grayscale for some calculations
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Brightness statistics
            brightness_mean = float(np.mean(gray))
            brightness_std = float(np.std(gray))
            
            # Contrast (standard deviation of pixel intensities)
            contrast = brightness_std / 255.0
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(np.var(laplacian)) / 10000.0  # Normalize
            
            # Dynamic range
            dynamic_range = (float(np.max(gray)) - float(np.min(gray))) / 255.0
            
            return {
                'brightness_mean': brightness_mean,
                'brightness_std': brightness_std,
                'contrast': contrast,
                'sharpness': sharpness,
                'dynamic_range': dynamic_range
            }
            
        except Exception as e:
            logger.warning(f"Error calculating image quality metrics: {e}")
            return {
                'brightness_mean': 0.0,
                'brightness_std': 0.0,
                'contrast': 0.0,
                'sharpness': 0.0,
                'dynamic_range': 0.0
            }
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics"""
        try:
            # Normalize and weight different quality aspects
            contrast_score = min(metrics['contrast'] * 2.0, 1.0)  # Weight contrast highly
            sharpness_score = min(metrics['sharpness'], 1.0)
            dynamic_range_score = metrics['dynamic_range']
            
            # Brightness should be reasonable (not too dark or too bright)
            brightness_score = 1.0 - abs(metrics['brightness_mean'] - 127.5) / 127.5
            brightness_score = max(0.0, brightness_score)
            
            # Weighted average
            weights = [0.3, 0.3, 0.2, 0.2]  # contrast, sharpness, dynamic_range, brightness
            scores = [contrast_score, sharpness_score, dynamic_range_score, brightness_score]
            
            quality_score = sum(w * s for w, s in zip(weights, scores))
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 0.0
    
    def validate_depth_map_quality(self, depth_map: np.ndarray) -> Dict[str, Any]:
        """
        Validate depth map quality using statistical measures
        """
        try:
            if depth_map is None or depth_map.size == 0:
                return {
                    'valid': False,
                    'score': 0.0,
                    'issues': ['Empty or None depth map'],
                    'metrics': {}
                }
            
            # Ensure depth map is 2D
            if len(depth_map.shape) != 2:
                return {
                    'valid': False,
                    'score': 0.0,
                    'issues': [f'Invalid depth map shape: {depth_map.shape}'],
                    'metrics': {}
                }
            
            # Calculate depth quality metrics
            depth_min = float(np.min(depth_map))
            depth_max = float(np.max(depth_map))
            depth_mean = float(np.mean(depth_map))
            depth_std = float(np.std(depth_map))
            
            # Dynamic range
            depth_range = depth_max - depth_min
            
            # Check for reasonable depth values
            issues = []
            if depth_range < 0.01:  # Very small range
                issues.append('Very small depth range (possibly flat/invalid)')
            
            if depth_std < 0.01:  # Very low variation
                issues.append('Very low depth variation')
            
            # Calculate gradient magnitude for edge detection
            grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            edge_strength = float(np.mean(gradient_magnitude))
            
            # Quality score based on variation and edge content
            variation_score = min(depth_std * 10.0, 1.0)  # Normalize std
            edge_score = min(edge_strength * 0.1, 1.0)  # Normalize edge strength
            range_score = min(depth_range * 2.0, 1.0)  # Normalize range
            
            quality_score = (variation_score + edge_score + range_score) / 3.0
            
            return {
                'valid': quality_score >= self.quality_threshold and len(issues) == 0,
                'score': quality_score,
                'issues': issues,
                'metrics': {
                    'depth_min': depth_min,
                    'depth_max': depth_max,
                    'depth_mean': depth_mean,
                    'depth_std': depth_std,
                    'depth_range': depth_range,
                    'edge_strength': edge_strength,
                    'shape': depth_map.shape
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating depth map: {e}")
            return {
                'valid': False,
                'score': 0.0,
                'issues': [f'Validation error: {str(e)}'],
                'metrics': {}
            }
    
    def compare_data_distributions(self, real_data: List[np.ndarray], 
                                 synthetic_data: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compare statistical distributions between real and synthetic data
        """
        try:
            if not real_data or not synthetic_data:
                return {
                    'valid': False,
                    'similarity_score': 0.0,
                    'issues': ['Empty data arrays'],
                    'metrics': {}
                }
            
            # Calculate statistics for both datasets
            real_stats = self._calculate_dataset_statistics(real_data)
            synthetic_stats = self._calculate_dataset_statistics(synthetic_data)
            
            # Compare distributions
            similarity_metrics = {}
            
            # Compare means
            mean_diff = abs(real_stats['mean'] - synthetic_stats['mean'])
            mean_similarity = max(0.0, 1.0 - mean_diff / 255.0)
            similarity_metrics['mean_similarity'] = mean_similarity
            
            # Compare standard deviations
            std_diff = abs(real_stats['std'] - synthetic_stats['std'])
            std_similarity = max(0.0, 1.0 - std_diff / 255.0)
            similarity_metrics['std_similarity'] = std_similarity
            
            # Compare dynamic ranges
            range_diff = abs(real_stats['range'] - synthetic_stats['range'])
            range_similarity = max(0.0, 1.0 - range_diff / 255.0)
            similarity_metrics['range_similarity'] = range_similarity
            
            # Overall similarity score
            similarity_score = (mean_similarity + std_similarity + range_similarity) / 3.0
            
            # Check for issues
            issues = []
            if similarity_score < 0.5:
                issues.append('Low similarity between real and synthetic data')
            if mean_diff > 50:
                issues.append('Large difference in mean brightness')
            if std_diff > 30:
                issues.append('Large difference in contrast/variation')
            
            return {
                'valid': similarity_score >= 0.6 and len(issues) == 0,
                'similarity_score': similarity_score,
                'issues': issues,
                'metrics': {
                    'real_stats': real_stats,
                    'synthetic_stats': synthetic_stats,
                    'similarity_metrics': similarity_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing data distributions: {e}")
            return {
                'valid': False,
                'similarity_score': 0.0,
                'issues': [f'Comparison error: {str(e)}'],
                'metrics': {}
            }
    
    def _calculate_dataset_statistics(self, data: List[np.ndarray]) -> Dict[str, float]:
        """Calculate statistical measures for a dataset"""
        try:
            all_values = []
            for item in data:
                if len(item.shape) == 3:  # Color image
                    gray = cv2.cvtColor(item, cv2.COLOR_RGB2GRAY)
                else:
                    gray = item
                all_values.extend(gray.flatten())
            
            all_values = np.array(all_values)
            
            return {
                'mean': float(np.mean(all_values)),
                'std': float(np.std(all_values)),
                'min': float(np.min(all_values)),
                'max': float(np.max(all_values)),
                'range': float(np.max(all_values) - np.min(all_values)),
                'count': len(data)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating dataset statistics: {e}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'count': 0
            }


# Convenience functions for direct use in notebooks
def validate_image_quality(image_path: str, quality_threshold: float = 0.7) -> Dict[str, Any]:
    """Validate image quality with comprehensive checks"""
    validator = DataValidator(quality_threshold)
    return validator.validate_image_quality(image_path)

def validate_depth_map_quality(depth_map: np.ndarray, quality_threshold: float = 0.7) -> Dict[str, Any]:
    """Validate depth map quality using statistical measures"""
    validator = DataValidator(quality_threshold)
    return validator.validate_depth_map_quality(depth_map)