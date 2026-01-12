"""
Enhanced Memory Management for FLUX.1-dev Pipeline
Provides dynamic batch sizing, GPU memory optimization, and quality validation
"""

import torch
import psutil
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import cv2
from scipy import stats
import logging
from dataclasses import dataclass
import time
import gc

@dataclass
class MemoryProfile:
    """Memory usage profile for monitoring"""
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_total_gb: float
    system_memory_percent: float
    timestamp: float

class FLUXMemoryManager:
    """Advanced memory management specifically for FLUX.1-dev operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_profiles: List[MemoryProfile] = []
        self.logger = logging.getLogger(__name__)
        
        # Memory thresholds
        self.gpu_memory_threshold = config.get('hardware', {}).get('gpu_memory_fraction', 0.9)
        self.system_memory_threshold = 85.0  # Percent
        
        # FLUX-specific memory estimates (GB per 1024x1024 image)
        self.flux_memory_estimates = {
            'base_model_loading': 8.0,  # FLUX.1-dev base model
            'controlnet_loading': 2.0,  # ControlNet model
            'generation_per_image': 2.5,  # Per image generation
            'batch_overhead': 0.5  # Additional overhead per batch
        }
    
    def get_current_memory_profile(self) -> MemoryProfile:
        """Get current memory usage profile"""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_allocated = gpu_reserved = gpu_total = 0.0
        
        system_memory = psutil.virtual_memory().percent
        
        profile = MemoryProfile(
            gpu_allocated_gb=gpu_allocated,
            gpu_reserved_gb=gpu_reserved,
            gpu_total_gb=gpu_total,
            system_memory_percent=system_memory,
            timestamp=time.time()
        )
        
        self.memory_profiles.append(profile)
        return profile
    
    def calculate_optimal_batch_size(self, image_resolution: Tuple[int, int] = (1024, 1024)) -> int:
        """Calculate optimal batch size based on available memory and image resolution"""
        profile = self.get_current_memory_profile()
        
        if not torch.cuda.is_available():
            return 1
        
        # Calculate available GPU memory
        available_memory = profile.gpu_total_gb - profile.gpu_allocated_gb
        
        # Adjust memory estimate based on image resolution
        resolution_factor = (image_resolution[0] * image_resolution[1]) / (1024 * 1024)
        memory_per_image = self.flux_memory_estimates['generation_per_image'] * resolution_factor
        
        # Calculate batch size with safety margin
        safety_margin = 0.8  # Use 80% of available memory
        max_batch_size = int((available_memory * safety_margin) / memory_per_image)
        
        # Ensure minimum batch size of 1
        batch_size = max(1, max_batch_size)
        
        # Log memory calculation
        self.logger.info(f"Memory calculation: Available={available_memory:.1f}GB, "
                        f"Per image={memory_per_image:.1f}GB, Batch size={batch_size}")
        
        return batch_size
    
    def monitor_memory_during_generation(self, operation_name: str = "generation") -> Dict[str, Any]:
        """Monitor memory usage during generation operations"""
        start_profile = self.get_current_memory_profile()
        
        return {
            'start_profile': start_profile,
            'operation_name': operation_name,
            'start_time': time.time()
        }
    
    def finalize_memory_monitoring(self, monitor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize memory monitoring and return usage statistics"""
        end_profile = self.get_current_memory_profile()
        start_profile = monitor_data['start_profile']
        
        memory_delta = {
            'gpu_allocated_delta': end_profile.gpu_allocated_gb - start_profile.gpu_allocated_gb,
            'gpu_reserved_delta': end_profile.gpu_reserved_gb - start_profile.gpu_reserved_gb,
            'system_memory_delta': end_profile.system_memory_percent - start_profile.system_memory_percent,
            'duration': end_profile.timestamp - start_profile.timestamp
        }
        
        return {
            'operation': monitor_data['operation_name'],
            'start_profile': start_profile,
            'end_profile': end_profile,
            'memory_delta': memory_delta,
            'peak_gpu_usage': end_profile.gpu_allocated_gb
        }
    
    def cleanup_memory(self, aggressive: bool = False):
        """Perform memory cleanup operations"""
        if aggressive:
            # Aggressive cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            # Standard cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Log memory state after cleanup
        profile = self.get_current_memory_profile()
        self.logger.info(f"Memory cleanup: GPU={profile.gpu_allocated_gb:.1f}GB, "
                        f"System={profile.system_memory_percent:.1f}%")
    
    def check_memory_health(self) -> Dict[str, Any]:
        """Check overall memory health and provide recommendations"""
        profile = self.get_current_memory_profile()
        
        health_status = {
            'gpu_usage_percent': (profile.gpu_allocated_gb / profile.gpu_total_gb * 100) if profile.gpu_total_gb > 0 else 0,
            'system_usage_percent': profile.system_memory_percent,
            'recommendations': [],
            'warnings': [],
            'status': 'healthy'
        }
        
        # Check GPU memory
        if health_status['gpu_usage_percent'] > 90:
            health_status['status'] = 'critical'
            health_status['warnings'].append('GPU memory usage critical (>90%)')
            health_status['recommendations'].append('Reduce batch size or enable CPU offloading')
        elif health_status['gpu_usage_percent'] > 75:
            health_status['status'] = 'warning'
            health_status['warnings'].append('GPU memory usage high (>75%)')
            health_status['recommendations'].append('Consider reducing batch size')
        
        # Check system memory
        if health_status['system_usage_percent'] > 90:
            health_status['status'] = 'critical'
            health_status['warnings'].append('System memory usage critical (>90%)')
            health_status['recommendations'].append('Close other applications or reduce processing load')
        elif health_status['system_usage_percent'] > 80:
            if health_status['status'] == 'healthy':
                health_status['status'] = 'warning'
            health_status['warnings'].append('System memory usage high (>80%)')
        
        return health_status

class QualityValidator:
    """Advanced quality validation for FLUX-generated synthetic images"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_thresholds = config.get('validation', {})
        self.logger = logging.getLogger(__name__)
    
    def validate_image_quality(self, image: Image.Image, depth_map: np.ndarray = None) -> Dict[str, Any]:
        """Comprehensive image quality validation"""
        img_array = np.array(image)
        
        # Basic quality metrics
        quality_metrics = {
            'resolution': image.size,
            'channels': len(img_array.shape),
            'brightness': self._calculate_brightness(img_array),
            'contrast': self._calculate_contrast(img_array),
            'sharpness': self._calculate_sharpness(img_array),
            'color_distribution': self._analyze_color_distribution(img_array),
            'noise_level': self._estimate_noise_level(img_array)
        }
        
        # Depth-specific validation if depth map provided
        if depth_map is not None:
            quality_metrics.update(self._validate_depth_consistency(img_array, depth_map))
        
        # Overall quality score
        quality_metrics['overall_score'] = self._calculate_overall_score(quality_metrics)
        
        # Quality assessment
        quality_metrics['assessment'] = self._assess_quality(quality_metrics)
        
        return quality_metrics
    
    def _calculate_brightness(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculate brightness metrics"""
        brightness = np.mean(img_array) / 255.0
        brightness_std = np.std(img_array) / 255.0
        
        return {
            'mean': float(brightness),
            'std': float(brightness_std),
            'is_acceptable': 0.1 <= brightness <= 0.9  # Avoid too dark or too bright
        }
    
    def _calculate_contrast(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculate contrast metrics"""
        # RMS contrast
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        contrast = np.std(gray) / 255.0
        
        # Michelson contrast
        max_val = np.max(gray)
        min_val = np.min(gray)
        michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-8)
        
        return {
            'rms_contrast': float(contrast),
            'michelson_contrast': float(michelson_contrast),
            'is_acceptable': contrast > 0.1  # Minimum contrast threshold
        }
    
    def _calculate_sharpness(self, img_array: np.ndarray) -> Dict[str, float]:
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        return {
            'laplacian_variance': float(sharpness),
            'is_acceptable': sharpness > 100  # Minimum sharpness threshold
        }
    
    def _analyze_color_distribution(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution properties"""
        if len(img_array.shape) != 3:
            return {'error': 'Not a color image'}
        
        # Calculate color statistics for each channel
        color_stats = {}
        for i, channel in enumerate(['red', 'green', 'blue']):
            channel_data = img_array[:, :, i]
            color_stats[channel] = {
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'range': float(np.max(channel_data) - np.min(channel_data))
            }
        
        # Color balance (how similar are the channel means)
        means = [color_stats[c]['mean'] for c in ['red', 'green', 'blue']]
        color_balance = 1.0 - (np.std(means) / np.mean(means)) if np.mean(means) > 0 else 0.0
        
        return {
            'channel_stats': color_stats,
            'color_balance': float(color_balance),
            'is_balanced': color_balance > 0.8  # Good color balance threshold
        }
    
    def _estimate_noise_level(self, img_array: np.ndarray) -> Dict[str, float]:
        """Estimate noise level in the image"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_estimate = np.var(laplacian)
        
        # Normalize noise estimate
        normalized_noise = noise_estimate / (255.0 ** 2)
        
        return {
            'noise_estimate': float(normalized_noise),
            'is_acceptable': normalized_noise < 0.1  # Low noise threshold
        }
    
    def _validate_depth_consistency(self, img_array: np.ndarray, depth_map: np.ndarray) -> Dict[str, Any]:
        """Validate consistency between image and depth map"""
        # Convert image to grayscale for edge detection
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect edges in both image and depth map
        img_edges = cv2.Canny(gray_img, 50, 150)
        depth_edges = cv2.Canny((depth_map * 255).astype(np.uint8), 50, 150)
        
        # Calculate edge correlation
        correlation = np.corrcoef(img_edges.flatten(), depth_edges.flatten())[0, 1]
        correlation = 0.0 if np.isnan(correlation) else correlation
        
        # Calculate depth-brightness correlation (closer objects should be brighter in many cases)
        brightness_map = np.mean(img_array, axis=2)
        depth_brightness_corr = np.corrcoef(depth_map.flatten(), brightness_map.flatten())[0, 1]
        depth_brightness_corr = 0.0 if np.isnan(depth_brightness_corr) else depth_brightness_corr
        
        return {
            'edge_correlation': float(correlation),
            'depth_brightness_correlation': float(abs(depth_brightness_corr)),
            'depth_consistency_score': float((abs(correlation) + abs(depth_brightness_corr)) / 2),
            'is_consistent': abs(correlation) > 0.3  # Minimum consistency threshold
        }
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics"""
        scores = []
        
        # Brightness score
        if 'brightness' in metrics and isinstance(metrics['brightness'], dict):
            brightness_score = 1.0 if metrics['brightness']['is_acceptable'] else 0.5
            scores.append(brightness_score)
        
        # Contrast score
        if 'contrast' in metrics and isinstance(metrics['contrast'], dict):
            contrast_score = 1.0 if metrics['contrast']['is_acceptable'] else 0.5
            scores.append(contrast_score)
        
        # Sharpness score
        if 'sharpness' in metrics and isinstance(metrics['sharpness'], dict):
            sharpness_score = 1.0 if metrics['sharpness']['is_acceptable'] else 0.5
            scores.append(sharpness_score)
        
        # Color distribution score
        if 'color_distribution' in metrics and isinstance(metrics['color_distribution'], dict):
            color_score = 1.0 if metrics['color_distribution'].get('is_balanced', False) else 0.7
            scores.append(color_score)
        
        # Noise score
        if 'noise_level' in metrics and isinstance(metrics['noise_level'], dict):
            noise_score = 1.0 if metrics['noise_level']['is_acceptable'] else 0.6
            scores.append(noise_score)
        
        # Depth consistency score (if available)
        if 'depth_consistency_score' in metrics:
            scores.append(metrics['depth_consistency_score'])
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _assess_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Provide quality assessment and recommendations"""
        overall_score = metrics.get('overall_score', 0.0)
        
        if overall_score >= 0.8:
            quality_level = 'excellent'
            recommendations = ['Image quality is excellent']
        elif overall_score >= 0.6:
            quality_level = 'good'
            recommendations = ['Image quality is good']
        elif overall_score >= 0.4:
            quality_level = 'acceptable'
            recommendations = ['Image quality is acceptable but could be improved']
        else:
            quality_level = 'poor'
            recommendations = ['Image quality is poor and should be regenerated']
        
        # Add specific recommendations based on metrics
        if 'brightness' in metrics and not metrics['brightness'].get('is_acceptable', True):
            recommendations.append('Adjust brightness levels')
        
        if 'contrast' in metrics and not metrics['contrast'].get('is_acceptable', True):
            recommendations.append('Improve image contrast')
        
        if 'sharpness' in metrics and not metrics['sharpness'].get('is_acceptable', True):
            recommendations.append('Increase image sharpness')
        
        return {
            'quality_level': quality_level,
            'score': overall_score,
            'recommendations': recommendations,
            'acceptable': overall_score >= 0.4
        }

class DataDistributionComparator:
    """Compare synthetic and real data distributions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compare_distributions(self, synthetic_images: List[Image.Image], 
                            real_images: List[Image.Image]) -> Dict[str, Any]:
        """Compare distributions between synthetic and real images"""
        
        # Extract features from both datasets
        synthetic_features = self._extract_features(synthetic_images, "synthetic")
        real_features = self._extract_features(real_images, "real")
        
        # Perform statistical comparisons
        comparison_results = {}
        
        for feature_name in synthetic_features.keys():
            if feature_name in real_features:
                comparison_results[feature_name] = self._compare_feature_distributions(
                    synthetic_features[feature_name],
                    real_features[feature_name],
                    feature_name
                )
        
        # Calculate overall similarity score
        similarity_scores = [comp.get('similarity_score', 0.0) for comp in comparison_results.values()]
        overall_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        return {
            'feature_comparisons': comparison_results,
            'overall_similarity': float(overall_similarity),
            'synthetic_count': len(synthetic_images),
            'real_count': len(real_images),
            'assessment': self._assess_distribution_similarity(overall_similarity)
        }
    
    def _extract_features(self, images: List[Image.Image], dataset_type: str) -> Dict[str, List[float]]:
        """Extract statistical features from images"""
        features = {
            'brightness': [],
            'contrast': [],
            'saturation': [],
            'hue_variance': [],
            'edge_density': []
        }
        
        self.logger.info(f"Extracting features from {len(images)} {dataset_type} images...")
        
        for img in images:
            img_array = np.array(img)
            
            # Brightness
            brightness = np.mean(img_array) / 255.0
            features['brightness'].append(brightness)
            
            # Contrast
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            contrast = np.std(gray) / 255.0
            features['contrast'].append(contrast)
            
            if len(img_array.shape) == 3:
                # Saturation
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv[:, :, 1]) / 255.0
                features['saturation'].append(saturation)
                
                # Hue variance
                hue_variance = np.var(hsv[:, :, 0]) / (180.0 ** 2)
                features['hue_variance'].append(hue_variance)
            else:
                features['saturation'].append(0.0)
                features['hue_variance'].append(0.0)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features['edge_density'].append(edge_density)
        
        return features
    
    def _compare_feature_distributions(self, synthetic_values: List[float], 
                                     real_values: List[float], 
                                     feature_name: str) -> Dict[str, Any]:
        """Compare distributions of a specific feature"""
        
        # Calculate basic statistics
        syn_stats = {
            'mean': np.mean(synthetic_values),
            'std': np.std(synthetic_values),
            'median': np.median(synthetic_values),
            'min': np.min(synthetic_values),
            'max': np.max(synthetic_values)
        }
        
        real_stats = {
            'mean': np.mean(real_values),
            'std': np.std(real_values),
            'median': np.median(real_values),
            'min': np.min(real_values),
            'max': np.max(real_values)
        }
        
        # Statistical tests
        try:
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(synthetic_values, real_values)
            
            # Mann-Whitney U test
            mw_statistic, mw_p_value = stats.mannwhitneyu(synthetic_values, real_values, alternative='two-sided')
            
            # Calculate similarity score based on mean difference
            mean_diff = abs(syn_stats['mean'] - real_stats['mean'])
            max_mean = max(syn_stats['mean'], real_stats['mean'], 0.001)
            similarity_score = 1.0 - min(1.0, mean_diff / max_mean)
            
        except Exception as e:
            self.logger.warning(f"Statistical test failed for {feature_name}: {e}")
            ks_statistic = ks_p_value = mw_statistic = mw_p_value = 0.0
            similarity_score = 0.0
        
        return {
            'synthetic_stats': syn_stats,
            'real_stats': real_stats,
            'ks_test': {'statistic': float(ks_statistic), 'p_value': float(ks_p_value)},
            'mannwhitney_test': {'statistic': float(mw_statistic), 'p_value': float(mw_p_value)},
            'similarity_score': float(similarity_score),
            'mean_difference': float(abs(syn_stats['mean'] - real_stats['mean']))
        }
    
    def _assess_distribution_similarity(self, overall_similarity: float) -> Dict[str, Any]:
        """Assess the overall distribution similarity"""
        if overall_similarity >= 0.9:
            assessment = 'excellent'
            message = 'Synthetic data distribution closely matches real data'
        elif overall_similarity >= 0.7:
            assessment = 'good'
            message = 'Synthetic data distribution is similar to real data'
        elif overall_similarity >= 0.5:
            assessment = 'acceptable'
            message = 'Synthetic data distribution has some differences from real data'
        else:
            assessment = 'poor'
            message = 'Synthetic data distribution significantly differs from real data'
        
        return {
            'level': assessment,
            'message': message,
            'score': overall_similarity,
            'acceptable': overall_similarity >= 0.5
        }