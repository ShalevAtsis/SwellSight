"""
Synthetic vs Real Data Comparison Framework for SwellSight Wave Analysis System.

This module provides tools for comparing synthetic and real datasets using statistical
methods, distribution matching validation, and perceptual quality assessment.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torchvision import transforms, models
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DistributionComparison:
    """Results of statistical distribution comparison between datasets."""
    kl_divergence: float
    wasserstein_distance: float
    statistical_tests: Dict[str, float]
    visual_similarity_score: float
    distribution_match_score: float
    feature_similarity: Dict[str, float]
    perceptual_distance: float


@dataclass
class DataDriftMetrics:
    """Metrics for detecting data drift between datasets."""
    drift_score: float
    drift_detected: bool
    drift_threshold: float
    affected_features: List[str]
    drift_magnitude: Dict[str, float]
    recommendations: List[str]


class PerceptualSimilarityCalculator:
    """
    Calculate perceptual similarity between images using pre-trained features.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize with pre-trained VGG features for perceptual similarity."""
        self.device = device
        
        # Load pre-trained VGG16 for feature extraction
        vgg = models.vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:23])  # Up to conv4_3
        self.feature_extractor.eval()
        self.feature_extractor.to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract perceptual features from a list of images."""
        features = []
        
        with torch.no_grad():
            for img in images:
                # Convert BGR to RGB if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img
                
                # Preprocess and extract features
                img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                feature_map = self.feature_extractor(img_tensor)
                
                # Global average pooling
                feature_vector = torch.mean(feature_map, dim=[2, 3]).cpu().numpy().flatten()
                features.append(feature_vector)
        
        return np.array(features)
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two sets of features."""
        # Calculate mean features for each dataset
        mean_features1 = np.mean(features1, axis=0)
        mean_features2 = np.mean(features2, axis=0)
        
        # Calculate cosine similarity
        dot_product = np.dot(mean_features1, mean_features2)
        norm1 = np.linalg.norm(mean_features1)
        norm2 = np.linalg.norm(mean_features2)
        
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        return similarity


class DatasetComparator:
    """
    Comprehensive comparison framework for synthetic and real datasets.
    
    This class provides statistical comparison tools, distribution matching validation,
    and perceptual quality assessment between datasets.
    """
    
    def __init__(self, 
                 drift_threshold: float = 0.1,
                 similarity_threshold: float = 0.8,
                 sample_size: int = 1000):
        """
        Initialize the dataset comparator.
        
        Args:
            drift_threshold: Threshold for detecting significant data drift
            similarity_threshold: Minimum similarity score for good match
            sample_size: Number of samples to use for comparison (for performance)
        """
        self.drift_threshold = drift_threshold
        self.similarity_threshold = similarity_threshold
        self.sample_size = sample_size
        self.perceptual_calculator = PerceptualSimilarityCalculator()
    
    def compare_datasets(self, 
                        real_dataset_path: Union[str, Path],
                        synthetic_dataset_path: Union[str, Path]) -> DistributionComparison:
        """
        Compare synthetic and real datasets comprehensively.
        
        Args:
            real_dataset_path: Path to real dataset
            synthetic_dataset_path: Path to synthetic dataset
            
        Returns:
            DistributionComparison: Comprehensive comparison results
        """
        logger.info("Loading datasets for comparison...")
        
        # Load image data
        real_images = self._load_images(Path(real_dataset_path))
        synthetic_images = self._load_images(Path(synthetic_dataset_path))
        
        # Sample for performance if datasets are large
        if len(real_images) > self.sample_size:
            real_images = np.random.choice(real_images, self.sample_size, replace=False)
        if len(synthetic_images) > self.sample_size:
            synthetic_images = np.random.choice(synthetic_images, self.sample_size, replace=False)
        
        logger.info(f"Comparing {len(real_images)} real vs {len(synthetic_images)} synthetic images")
        
        # Extract statistical features
        real_features = self._extract_statistical_features(real_images)
        synthetic_features = self._extract_statistical_features(synthetic_images)
        
        # Calculate statistical comparisons
        kl_div = self._calculate_kl_divergence(real_features, synthetic_features)
        wasserstein_dist = self._calculate_wasserstein_distance(real_features, synthetic_features)
        statistical_tests = self._perform_statistical_tests(real_features, synthetic_features)
        
        # Calculate perceptual similarity
        logger.info("Calculating perceptual similarity...")
        real_perceptual = self.perceptual_calculator.extract_features([cv2.imread(str(p)) for p in real_images[:50]])
        synthetic_perceptual = self.perceptual_calculator.extract_features([cv2.imread(str(p)) for p in synthetic_images[:50]])
        visual_similarity = self.perceptual_calculator.calculate_similarity(real_perceptual, synthetic_perceptual)
        
        # Calculate feature-wise similarity
        feature_similarity = self._calculate_feature_similarity(real_features, synthetic_features)
        
        # Calculate overall distribution match score
        distribution_match_score = self._calculate_distribution_match_score(
            kl_div, wasserstein_dist, visual_similarity, statistical_tests
        )
        
        # Calculate perceptual distance
        perceptual_distance = 1.0 - visual_similarity
        
        return DistributionComparison(
            kl_divergence=kl_div,
            wasserstein_distance=wasserstein_dist,
            statistical_tests=statistical_tests,
            visual_similarity_score=visual_similarity,
            distribution_match_score=distribution_match_score,
            feature_similarity=feature_similarity,
            perceptual_distance=perceptual_distance
        )
    
    def detect_data_drift(self, 
                         baseline_dataset_path: Union[str, Path],
                         current_dataset_path: Union[str, Path]) -> DataDriftMetrics:
        """
        Detect data drift between baseline and current datasets.
        
        Args:
            baseline_dataset_path: Path to baseline dataset
            current_dataset_path: Path to current dataset
            
        Returns:
            DataDriftMetrics: Data drift detection results
        """
        logger.info("Detecting data drift...")
        
        # Load datasets
        baseline_images = self._load_images(Path(baseline_dataset_path))
        current_images = self._load_images(Path(current_dataset_path))
        
        # Extract features
        baseline_features = self._extract_statistical_features(baseline_images[:self.sample_size])
        current_features = self._extract_statistical_features(current_images[:self.sample_size])
        
        # Calculate drift metrics
        drift_scores = {}
        affected_features = []
        
        feature_names = ['brightness', 'contrast', 'saturation', 'sharpness', 'texture']
        
        for i, feature_name in enumerate(feature_names):
            if i < baseline_features.shape[1] and i < current_features.shape[1]:
                # Kolmogorov-Smirnov test for distribution difference
                ks_stat, p_value = ks_2samp(baseline_features[:, i], current_features[:, i])
                drift_scores[feature_name] = ks_stat
                
                if ks_stat > self.drift_threshold:
                    affected_features.append(feature_name)
        
        # Overall drift score
        overall_drift_score = np.mean(list(drift_scores.values()))
        drift_detected = overall_drift_score > self.drift_threshold
        
        # Generate recommendations
        recommendations = self._generate_drift_recommendations(drift_detected, affected_features)
        
        return DataDriftMetrics(
            drift_score=overall_drift_score,
            drift_detected=drift_detected,
            drift_threshold=self.drift_threshold,
            affected_features=affected_features,
            drift_magnitude=drift_scores,
            recommendations=recommendations
        )
    
    def _load_images(self, dataset_path: Path) -> List[Path]:
        """Load image file paths from dataset directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _extract_statistical_features(self, image_paths: List[Path]) -> np.ndarray:
        """Extract statistical features from images for comparison."""
        features = []
        
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Convert to different color spaces
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Extract various statistical features
                feature_vector = []
                
                # Brightness (mean intensity)
                brightness = np.mean(gray)
                feature_vector.append(brightness)
                
                # Contrast (standard deviation)
                contrast = np.std(gray)
                feature_vector.append(contrast)
                
                # Saturation (mean saturation in HSV)
                saturation = np.mean(hsv[:, :, 1])
                feature_vector.append(saturation)
                
                # Sharpness (Laplacian variance)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                feature_vector.append(sharpness)
                
                # Texture (Local Binary Pattern approximation)
                # Simplified texture measure using gradient magnitude
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                texture = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                feature_vector.append(texture)
                
                # Color distribution (histogram features)
                hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
                hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
                hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
                
                # Add histogram statistics
                feature_vector.extend([
                    np.mean(hist_b), np.std(hist_b),
                    np.mean(hist_g), np.std(hist_g),
                    np.mean(hist_r), np.std(hist_r)
                ])
                
                features.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Error extracting features from {img_path}: {e}")
                continue
        
        return np.array(features)
    
    def _calculate_kl_divergence(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate KL divergence between feature distributions."""
        try:
            # Calculate histograms for each feature dimension
            kl_divs = []
            
            for i in range(features1.shape[1]):
                # Create histograms
                hist1, bins = np.histogram(features1[:, i], bins=50, density=True)
                hist2, _ = np.histogram(features2[:, i], bins=bins, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                hist1 = hist1 + epsilon
                hist2 = hist2 + epsilon
                
                # Normalize
                hist1 = hist1 / np.sum(hist1)
                hist2 = hist2 / np.sum(hist2)
                
                # Calculate KL divergence
                kl_div = np.sum(hist1 * np.log(hist1 / hist2))
                
                # Ensure non-negative and finite
                if np.isfinite(kl_div) and kl_div >= 0:
                    kl_divs.append(kl_div)
                else:
                    kl_divs.append(0.0)  # Default to 0 for invalid values
            
            return max(0.0, np.mean(kl_divs)) if kl_divs else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating KL divergence: {e}")
            return 0.0  # Return 0 instead of inf on error
    
    def _calculate_wasserstein_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate Wasserstein distance between feature distributions."""
        try:
            distances = []
            
            for i in range(features1.shape[1]):
                dist = wasserstein_distance(features1[:, i], features2[:, i])
                distances.append(dist)
            
            return np.mean(distances)
            
        except Exception as e:
            logger.warning(f"Error calculating Wasserstein distance: {e}")
            return float('inf')
    
    def _perform_statistical_tests(self, features1: np.ndarray, features2: np.ndarray) -> Dict[str, float]:
        """Perform various statistical tests to compare distributions."""
        tests = {}
        
        try:
            # Kolmogorov-Smirnov test
            ks_stats = []
            for i in range(features1.shape[1]):
                ks_stat, p_value = ks_2samp(features1[:, i], features2[:, i])
                ks_stats.append(ks_stat)
            tests['ks_statistic'] = np.mean(ks_stats)
            
            # Mann-Whitney U test
            mw_stats = []
            for i in range(features1.shape[1]):
                try:
                    mw_stat, p_value = mannwhitneyu(features1[:, i], features2[:, i])
                    mw_stats.append(mw_stat)
                except ValueError:
                    # Handle case where all values are identical
                    mw_stats.append(0.5)
            tests['mannwhitney_u'] = np.mean(mw_stats)
            
            # T-test for means
            t_stats = []
            for i in range(features1.shape[1]):
                t_stat, p_value = stats.ttest_ind(features1[:, i], features2[:, i])
                t_stats.append(abs(t_stat))
            tests['t_statistic'] = np.mean(t_stats)
            
        except Exception as e:
            logger.warning(f"Error in statistical tests: {e}")
            tests = {'ks_statistic': 1.0, 'mannwhitney_u': 0.0, 't_statistic': 10.0}
        
        return tests
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> Dict[str, float]:
        """Calculate similarity for each feature dimension."""
        feature_names = ['brightness', 'contrast', 'saturation', 'sharpness', 'texture']
        similarities = {}
        
        for i, name in enumerate(feature_names):
            if i < features1.shape[1]:
                # Calculate correlation coefficient
                try:
                    corr, _ = stats.pearsonr(
                        np.mean(features1[:, i:i+1], axis=0),
                        np.mean(features2[:, i:i+1], axis=0)
                    )
                    similarities[name] = abs(corr) if not np.isnan(corr) else 0.0
                except:
                    similarities[name] = 0.0
        
        return similarities
    
    def _calculate_distribution_match_score(self, 
                                          kl_div: float,
                                          wasserstein_dist: float,
                                          visual_similarity: float,
                                          statistical_tests: Dict[str, float]) -> float:
        """Calculate overall distribution match score (0-1)."""
        # Normalize and invert KL divergence (lower is better)
        kl_score = max(0, 1 - min(kl_div / 2.0, 1.0))
        
        # Normalize and invert Wasserstein distance (lower is better)
        wasserstein_score = max(0, 1 - min(wasserstein_dist / 100.0, 1.0))
        
        # Visual similarity is already 0-1 (higher is better)
        visual_score = visual_similarity
        
        # Statistical test score (lower KS statistic is better)
        ks_score = max(0, 1 - min(statistical_tests.get('ks_statistic', 1.0), 1.0))
        
        # Weighted average
        overall_score = (
            0.3 * kl_score +
            0.3 * wasserstein_score +
            0.3 * visual_score +
            0.1 * ks_score
        )
        
        return overall_score
    
    def _generate_drift_recommendations(self, drift_detected: bool, affected_features: List[str]) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        
        if not drift_detected:
            recommendations.append("No significant data drift detected - dataset is stable")
            return recommendations
        
        recommendations.append("Significant data drift detected - investigate data collection pipeline")
        
        if 'brightness' in affected_features:
            recommendations.append("Brightness distribution has changed - check lighting conditions or camera settings")
        
        if 'contrast' in affected_features:
            recommendations.append("Contrast distribution has changed - verify image preprocessing pipeline")
        
        if 'saturation' in affected_features:
            recommendations.append("Color saturation has changed - check camera color calibration")
        
        if 'sharpness' in affected_features:
            recommendations.append("Image sharpness has changed - verify camera focus and image quality")
        
        if 'texture' in affected_features:
            recommendations.append("Texture patterns have changed - investigate scene composition changes")
        
        recommendations.append("Consider retraining models with updated data distribution")
        recommendations.append("Implement continuous monitoring for early drift detection")
        
        return recommendations


def create_comparison_visualization(comparison: DistributionComparison,
                                  real_features: np.ndarray,
                                  synthetic_features: np.ndarray,
                                  output_path: Optional[str] = None) -> None:
    """
    Create visualizations for dataset comparison results.
    
    Args:
        comparison: Comparison results
        real_features: Real dataset features
        synthetic_features: Synthetic dataset features
        output_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Dataset Comparison (Match Score: {comparison.distribution_match_score:.3f})', fontsize=16)
    
    feature_names = ['Brightness', 'Contrast', 'Saturation', 'Sharpness', 'Texture']
    
    # Feature distributions
    for i in range(min(5, real_features.shape[1])):
        row = i // 3
        col = i % 3
        if row < 2 and col < 3:
            axes[row, col].hist(real_features[:, i], alpha=0.7, label='Real', bins=30, density=True)
            axes[row, col].hist(synthetic_features[:, i], alpha=0.7, label='Synthetic', bins=30, density=True)
            axes[row, col].set_title(f'{feature_names[i]} Distribution')
            axes[row, col].legend()
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Density')
    
    # Summary metrics in the last subplot
    if axes.shape[0] > 1 and axes.shape[1] > 2:
        metrics = [
            f"KL Divergence: {comparison.kl_divergence:.3f}",
            f"Wasserstein Distance: {comparison.wasserstein_distance:.3f}",
            f"Visual Similarity: {comparison.visual_similarity_score:.3f}",
            f"Distribution Match: {comparison.distribution_match_score:.3f}",
            f"Perceptual Distance: {comparison.perceptual_distance:.3f}"
        ]
        
        axes[1, 2].text(0.1, 0.9, '\n'.join(metrics), transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Comparison Metrics')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_drift_visualization(drift_metrics: DataDriftMetrics,
                             output_path: Optional[str] = None) -> None:
    """
    Create visualizations for data drift detection results.
    
    Args:
        drift_metrics: Drift detection results
        output_path: Optional path to save the visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Drift scores by feature
    features = list(drift_metrics.drift_magnitude.keys())
    scores = list(drift_metrics.drift_magnitude.values())
    
    colors = ['red' if score > drift_metrics.drift_threshold else 'green' for score in scores]
    
    ax1.bar(features, scores, color=colors, alpha=0.7)
    ax1.axhline(y=drift_metrics.drift_threshold, color='red', linestyle='--', 
                label=f'Drift Threshold ({drift_metrics.drift_threshold})')
    ax1.set_title('Data Drift by Feature')
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Drift Score')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Overall drift status
    status_text = "DRIFT DETECTED" if drift_metrics.drift_detected else "NO DRIFT"
    status_color = "red" if drift_metrics.drift_detected else "green"
    
    ax2.text(0.5, 0.7, status_text, ha='center', va='center', 
             fontsize=20, fontweight='bold', color=status_color,
             transform=ax2.transAxes)
    
    ax2.text(0.5, 0.5, f"Overall Score: {drift_metrics.drift_score:.3f}", 
             ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    
    if drift_metrics.affected_features:
        affected_text = "Affected Features:\n" + "\n".join(drift_metrics.affected_features)
        ax2.text(0.5, 0.3, affected_text, ha='center', va='center', 
                 fontsize=12, transform=ax2.transAxes)
    
    ax2.set_title('Drift Detection Summary')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()