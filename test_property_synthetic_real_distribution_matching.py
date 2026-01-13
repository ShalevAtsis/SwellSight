"""
Property-based tests for synthetic vs real data distribution matching.

Tests Property 27: Synthetic-Real Distribution Matching
Validates Requirements 6.5
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, assume
from typing import List, Tuple, Dict
import logging

from src.swellsight.evaluation.data_comparison import (
    DatasetComparator, DistributionComparison, DataDriftMetrics
)

logger = logging.getLogger(__name__)


class TestSyntheticRealDistributionMatching:
    """Property-based tests for synthetic vs real data distribution matching."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = DatasetComparator(
            drift_threshold=0.1,
            similarity_threshold=0.8,
            sample_size=50  # Smaller sample for faster tests
        )
        self.temp_dirs = []
    
    def teardown_method(self):
        """Clean up test fixtures."""
        for temp_dir in self.temp_dirs:
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
    
    def create_synthetic_dataset(self, 
                                num_images: int,
                                distribution_params: Dict[str, float],
                                base_characteristics: str = "ocean") -> Path:
        """Create a synthetic dataset with specified distribution parameters."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        dataset_path = Path(temp_dir) / "synthetic_dataset"
        dataset_path.mkdir(exist_ok=True)
        
        for i in range(num_images):
            img = self.create_synthetic_image(distribution_params, base_characteristics)
            filename = f"synthetic_{i:04d}.jpg"
            filepath = dataset_path / filename
            cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return dataset_path
    
    def create_real_dataset(self, 
                           num_images: int,
                           distribution_params: Dict[str, float],
                           variation_factor: float = 0.1) -> Path:
        """Create a 'real' dataset that may or may not match synthetic distribution."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        dataset_path = Path(temp_dir) / "real_dataset"
        dataset_path.mkdir(exist_ok=True)
        
        # Add variation to distribution parameters to simulate real vs synthetic differences
        varied_params = {}
        for key, value in distribution_params.items():
            if isinstance(value, (int, float)):
                variation = np.random.normal(0, variation_factor * abs(value))
                varied_params[key] = max(0, value + variation)
            else:
                varied_params[key] = value
        
        for i in range(num_images):
            img = self.create_synthetic_image(varied_params, "ocean")
            filename = f"real_{i:04d}.jpg"
            filepath = dataset_path / filename
            cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Slightly different quality
        
        return dataset_path
    
    def create_synthetic_image(self, 
                              distribution_params: Dict[str, float],
                              base_type: str = "ocean") -> np.ndarray:
        """Create a synthetic image with specified characteristics."""
        width, height = 512, 384  # Fixed size for consistency
        
        # Base image generation
        if base_type == "ocean":
            # Ocean-like base colors (blue-green spectrum)
            base_blue = int(distribution_params.get('base_blue', 120))
            base_green = int(distribution_params.get('base_green', 80))
            base_red = int(distribution_params.get('base_red', 40))
            
            img = np.full((height, width, 3), [base_blue, base_green, base_red], dtype=np.uint8)
        else:
            # Generic base
            base_intensity = int(distribution_params.get('base_intensity', 128))
            img = np.full((height, width, 3), base_intensity, dtype=np.uint8)
        
        # Add controlled noise and patterns
        brightness_std = distribution_params.get('brightness_std', 30)
        contrast_factor = distribution_params.get('contrast_factor', 1.0)
        saturation_factor = distribution_params.get('saturation_factor', 1.0)
        
        # Brightness variation
        brightness_noise = np.random.normal(0, brightness_std, (height, width, 3))
        img = np.clip(img.astype(float) + brightness_noise, 0, 255).astype(np.uint8)
        
        # Contrast adjustment
        img = np.clip((img.astype(float) - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # Saturation adjustment (convert to HSV, modify S channel)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(float) * saturation_factor, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add texture patterns
        texture_strength = distribution_params.get('texture_strength', 0.1)
        if texture_strength > 0:
            # Simple wave-like texture
            x = np.arange(width)
            y = np.arange(height)
            X, Y = np.meshgrid(x, y)
            wave_pattern = np.sin(X * 0.1) * np.sin(Y * 0.05) * texture_strength * 50
            
            for c in range(3):
                img[:, :, c] = np.clip(img[:, :, c].astype(float) + wave_pattern, 0, 255).astype(np.uint8)
        
        return img
    
    @given(
        num_images=st.integers(min_value=10, max_value=30),
        distribution_similarity=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=8, deadline=45000)
    def test_distribution_matching_correlation(self, num_images: int, distribution_similarity: float):
        """
        Property: Distribution matching score should correlate with actual similarity.
        
        When synthetic and real datasets have similar distributions, the matching
        score should be high. When they differ significantly, the score should be low.
        """
        # Create base distribution parameters
        base_params = {
            'base_blue': 120,
            'base_green': 80,
            'base_red': 40,
            'brightness_std': 25,
            'contrast_factor': 1.0,
            'saturation_factor': 1.0,
            'texture_strength': 0.1
        }
        
        # Create synthetic dataset
        synthetic_path = self.create_synthetic_dataset(num_images, base_params)
        
        # Create real dataset with controlled similarity
        variation_factor = (1.0 - distribution_similarity) * 0.5  # Scale variation inversely with similarity
        real_path = self.create_real_dataset(num_images, base_params, variation_factor)
        
        try:
            comparison = self.comparator.compare_datasets(real_path, synthetic_path)
            
            # Distribution match score should correlate with intended similarity
            if distribution_similarity > 0.8:
                assert comparison.distribution_match_score > 0.5, f"High similarity ({distribution_similarity:.2f}) should yield high match score, got {comparison.distribution_match_score:.3f}"
            elif distribution_similarity < 0.3:
                assert comparison.distribution_match_score < 0.8, f"Low similarity ({distribution_similarity:.2f}) should yield lower match score, got {comparison.distribution_match_score:.3f}"
            
            # All metrics should be within valid ranges
            assert 0 <= comparison.distribution_match_score <= 1, f"Invalid distribution match score: {comparison.distribution_match_score}"
            assert 0 <= comparison.visual_similarity_score <= 1, f"Invalid visual similarity score: {comparison.visual_similarity_score}"
            assert comparison.kl_divergence >= 0, f"Invalid KL divergence: {comparison.kl_divergence}"
            assert comparison.wasserstein_distance >= 0, f"Invalid Wasserstein distance: {comparison.wasserstein_distance}"
            
            logger.info(f"✓ Distribution matching correlation verified: {distribution_similarity:.2f} similarity → {comparison.distribution_match_score:.3f} match score")
            
        except Exception as e:
            pytest.fail(f"Distribution matching correlation test failed: {e}")
    
    @given(
        num_images=st.integers(min_value=8, max_value=20),
        brightness_difference=st.floats(min_value=0.0, max_value=100.0),
        contrast_difference=st.floats(min_value=0.0, max_value=2.0)
    )
    @settings(max_examples=8, deadline=45000)
    def test_statistical_distance_sensitivity(self, num_images: int, brightness_difference: float, contrast_difference: float):
        """
        Property: Statistical distance metrics should be sensitive to distribution differences.
        
        Larger differences in brightness and contrast should result in larger
        statistical distances (KL divergence, Wasserstein distance).
        """
        # Create base parameters
        base_params = {
            'base_blue': 120,
            'base_green': 80,
            'base_red': 40,
            'brightness_std': 25,
            'contrast_factor': 1.0,
            'saturation_factor': 1.0,
            'texture_strength': 0.1
        }
        
        # Create synthetic dataset
        synthetic_path = self.create_synthetic_dataset(num_images, base_params)
        
        # Create real dataset with specific differences
        different_params = base_params.copy()
        different_params['brightness_std'] = base_params['brightness_std'] + brightness_difference
        different_params['contrast_factor'] = base_params['contrast_factor'] + contrast_difference
        
        real_path = self.create_synthetic_dataset(num_images, different_params)
        
        try:
            comparison = self.comparator.compare_datasets(real_path, synthetic_path)
            
            # Statistical distances should increase with parameter differences
            total_difference = brightness_difference + contrast_difference
            
            if total_difference > 50:  # Significant difference
                assert comparison.kl_divergence > 0.1, f"Large parameter difference ({total_difference:.1f}) should yield higher KL divergence, got {comparison.kl_divergence:.3f}"
                assert comparison.wasserstein_distance > 1.0, f"Large parameter difference should yield higher Wasserstein distance, got {comparison.wasserstein_distance:.3f}"
            
            # Metrics should be non-negative and finite
            assert np.isfinite(comparison.kl_divergence), f"KL divergence should be finite: {comparison.kl_divergence}"
            assert np.isfinite(comparison.wasserstein_distance), f"Wasserstein distance should be finite: {comparison.wasserstein_distance}"
            assert comparison.kl_divergence >= 0, f"KL divergence should be non-negative: {comparison.kl_divergence}"
            assert comparison.wasserstein_distance >= 0, f"Wasserstein distance should be non-negative: {comparison.wasserstein_distance}"
            
            logger.info(f"✓ Statistical distance sensitivity verified: {total_difference:.1f} param diff → KL: {comparison.kl_divergence:.3f}, Wasserstein: {comparison.wasserstein_distance:.3f}")
            
        except Exception as e:
            pytest.fail(f"Statistical distance sensitivity test failed: {e}")
    
    @given(
        num_images=st.integers(min_value=10, max_value=25),
        feature_variation=st.floats(min_value=0.0, max_value=0.5)
    )
    @settings(max_examples=6, deadline=45000)
    def test_feature_similarity_consistency(self, num_images: int, feature_variation: float):
        """
        Property: Feature similarity should be consistent across multiple runs.
        
        Running the same comparison multiple times should yield similar results.
        """
        # Create datasets
        base_params = {
            'base_blue': 100,
            'base_green': 70,
            'base_red': 30,
            'brightness_std': 20,
            'contrast_factor': 1.2,
            'saturation_factor': 0.9,
            'texture_strength': 0.15
        }
        
        synthetic_path = self.create_synthetic_dataset(num_images, base_params)
        
        # Create real dataset with controlled variation
        varied_params = {}
        for key, value in base_params.items():
            if isinstance(value, (int, float)):
                varied_params[key] = value * (1 + feature_variation)
            else:
                varied_params[key] = value
        
        real_path = self.create_synthetic_dataset(num_images, varied_params)
        
        try:
            # Run comparison multiple times
            comparison1 = self.comparator.compare_datasets(real_path, synthetic_path)
            comparison2 = self.comparator.compare_datasets(real_path, synthetic_path)
            
            # Results should be consistent (allowing for small numerical differences)
            tolerance = 0.05
            
            assert abs(comparison1.distribution_match_score - comparison2.distribution_match_score) < tolerance, \
                f"Distribution match scores should be consistent: {comparison1.distribution_match_score:.3f} vs {comparison2.distribution_match_score:.3f}"
            
            assert abs(comparison1.visual_similarity_score - comparison2.visual_similarity_score) < tolerance, \
                f"Visual similarity scores should be consistent: {comparison1.visual_similarity_score:.3f} vs {comparison2.visual_similarity_score:.3f}"
            
            # Feature similarities should also be consistent
            for feature_name in comparison1.feature_similarity:
                if feature_name in comparison2.feature_similarity:
                    diff = abs(comparison1.feature_similarity[feature_name] - comparison2.feature_similarity[feature_name])
                    assert diff < tolerance, f"Feature similarity for {feature_name} should be consistent: {diff:.3f} > {tolerance}"
            
            logger.info(f"✓ Feature similarity consistency verified across multiple runs")
            
        except Exception as e:
            pytest.fail(f"Feature similarity consistency test failed: {e}")
    
    @given(
        num_images=st.integers(min_value=8, max_value=20)
    )
    @settings(max_examples=6, deadline=45000)
    def test_identical_dataset_comparison(self, num_images: int):
        """
        Property: Comparing identical datasets should yield perfect similarity.
        
        When comparing a dataset to itself, all similarity metrics should
        indicate perfect or near-perfect match.
        """
        # Create dataset
        base_params = {
            'base_blue': 110,
            'base_green': 75,
            'base_red': 35,
            'brightness_std': 30,
            'contrast_factor': 1.1,
            'saturation_factor': 1.0,
            'texture_strength': 0.1
        }
        
        dataset_path = self.create_synthetic_dataset(num_images, base_params)
        
        try:
            # Compare dataset to itself
            comparison = self.comparator.compare_datasets(dataset_path, dataset_path)
            
            # Should have perfect or near-perfect similarity
            assert comparison.distribution_match_score > 0.95, f"Identical datasets should have high match score, got {comparison.distribution_match_score:.3f}"
            assert comparison.visual_similarity_score > 0.95, f"Identical datasets should have high visual similarity, got {comparison.visual_similarity_score:.3f}"
            
            # Statistical distances should be very small
            assert comparison.kl_divergence < 0.1, f"Identical datasets should have low KL divergence, got {comparison.kl_divergence:.3f}"
            assert comparison.wasserstein_distance < 1.0, f"Identical datasets should have low Wasserstein distance, got {comparison.wasserstein_distance:.3f}"
            
            # Perceptual distance should be very small
            assert comparison.perceptual_distance < 0.1, f"Identical datasets should have low perceptual distance, got {comparison.perceptual_distance:.3f}"
            
            logger.info(f"✓ Identical dataset comparison verified: match score {comparison.distribution_match_score:.3f}")
            
        except Exception as e:
            pytest.fail(f"Identical dataset comparison test failed: {e}")
    
    @given(
        num_images=st.integers(min_value=10, max_value=20),
        drift_magnitude=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=6, deadline=45000)
    def test_data_drift_detection_sensitivity(self, num_images: int, drift_magnitude: float):
        """
        Property: Data drift detection should be sensitive to distribution changes.
        
        Larger distribution changes should result in higher drift scores and
        more likely drift detection.
        """
        # Create baseline dataset
        baseline_params = {
            'base_blue': 120,
            'base_green': 80,
            'base_red': 40,
            'brightness_std': 25,
            'contrast_factor': 1.0,
            'saturation_factor': 1.0,
            'texture_strength': 0.1
        }
        
        baseline_path = self.create_synthetic_dataset(num_images, baseline_params)
        
        # Create current dataset with controlled drift
        current_params = baseline_params.copy()
        current_params['brightness_std'] = baseline_params['brightness_std'] * (1 + drift_magnitude)
        current_params['contrast_factor'] = baseline_params['contrast_factor'] * (1 + drift_magnitude * 0.5)
        
        current_path = self.create_synthetic_dataset(num_images, current_params)
        
        try:
            drift_metrics = self.comparator.detect_data_drift(baseline_path, current_path)
            
            # Drift score should correlate with drift magnitude
            if drift_magnitude > 0.5:
                assert drift_metrics.drift_score > 0.05, f"High drift magnitude ({drift_magnitude:.2f}) should yield higher drift score, got {drift_metrics.drift_score:.3f}"
                # May or may not detect drift depending on threshold, but score should be elevated
            
            if drift_magnitude < 0.1:
                assert drift_metrics.drift_score < 0.2, f"Low drift magnitude ({drift_magnitude:.2f}) should yield lower drift score, got {drift_metrics.drift_score:.3f}"
            
            # Drift score should be non-negative and finite
            assert drift_metrics.drift_score >= 0, f"Drift score should be non-negative: {drift_metrics.drift_score}"
            assert np.isfinite(drift_metrics.drift_score), f"Drift score should be finite: {drift_metrics.drift_score}"
            
            # Drift threshold should be consistent
            assert drift_metrics.drift_threshold == self.comparator.drift_threshold, "Drift threshold should match comparator setting"
            
            # Affected features should be reasonable
            assert isinstance(drift_metrics.affected_features, list), "Affected features should be a list"
            assert len(drift_metrics.affected_features) <= 10, "Should not have too many affected features"
            
            logger.info(f"✓ Data drift detection sensitivity verified: {drift_magnitude:.2f} magnitude → {drift_metrics.drift_score:.3f} drift score, detected: {drift_metrics.drift_detected}")
            
        except Exception as e:
            pytest.fail(f"Data drift detection sensitivity test failed: {e}")
    
    @given(
        num_images=st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=5, deadline=30000)
    def test_statistical_test_validity(self, num_images: int):
        """
        Property: Statistical tests should produce valid results.
        
        All statistical test results should be within expected ranges and
        should be mathematically valid.
        """
        # Create two different datasets
        params1 = {
            'base_blue': 100,
            'base_green': 60,
            'base_red': 30,
            'brightness_std': 20,
            'contrast_factor': 1.0,
            'saturation_factor': 1.0,
            'texture_strength': 0.1
        }
        
        params2 = {
            'base_blue': 130,
            'base_green': 90,
            'base_red': 50,
            'brightness_std': 35,
            'contrast_factor': 1.3,
            'saturation_factor': 0.8,
            'texture_strength': 0.2
        }
        
        dataset1_path = self.create_synthetic_dataset(num_images, params1)
        dataset2_path = self.create_synthetic_dataset(num_images, params2)
        
        try:
            comparison = self.comparator.compare_datasets(dataset1_path, dataset2_path)
            
            # Check statistical test results
            statistical_tests = comparison.statistical_tests
            
            # KS statistic should be between 0 and 1
            if 'ks_statistic' in statistical_tests:
                ks_stat = statistical_tests['ks_statistic']
                assert 0 <= ks_stat <= 1, f"KS statistic should be between 0 and 1: {ks_stat}"
                assert np.isfinite(ks_stat), f"KS statistic should be finite: {ks_stat}"
            
            # Mann-Whitney U statistic should be reasonable
            if 'mannwhitney_u' in statistical_tests:
                mw_stat = statistical_tests['mannwhitney_u']
                assert mw_stat >= 0, f"Mann-Whitney U statistic should be non-negative: {mw_stat}"
                assert np.isfinite(mw_stat), f"Mann-Whitney U statistic should be finite: {mw_stat}"
            
            # T-statistic should be finite
            if 't_statistic' in statistical_tests:
                t_stat = statistical_tests['t_statistic']
                assert np.isfinite(t_stat), f"T-statistic should be finite: {t_stat}"
                assert t_stat >= 0, f"Absolute T-statistic should be non-negative: {t_stat}"
            
            logger.info(f"✓ Statistical test validity verified: {statistical_tests}")
            
        except Exception as e:
            pytest.fail(f"Statistical test validity test failed: {e}")


if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v", "--tb=short"])