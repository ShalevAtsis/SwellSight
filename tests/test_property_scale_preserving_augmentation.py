"""
Property-based test for scale-preserving augmentation.

Tests Property 23: Scale-Preserving Augmentation
Validates Requirements 9.3, 9.4
"""

import pytest
import numpy as np
import cv2
from hypothesis import given, strategies as st, settings, assume
from typing import Tuple

from src.swellsight.data.augmentation import WaveAugmentation, AugmentationConfig


class TestScalePreservingAugmentation:
    """Property-based tests for scale-preserving augmentation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = AugmentationConfig(
            enable_weather_effects=True,
            enable_lighting_changes=True,
            preserve_scale=True,
            scale_tolerance=0.05,
            edge_preservation_threshold=0.8
        )
        self.augmentation = WaveAugmentation(self.config)
    
    @given(
        height=st.integers(min_value=256, max_value=1024),
        width=st.integers(min_value=256, max_value=1024),
        brightness=st.floats(min_value=50, max_value=200),
        contrast=st.floats(min_value=0.5, max_value=2.0)
    )
    @settings(max_examples=50, deadline=5000)
    def test_property_scale_preserving_augmentation(self, height: int, width: int, 
                                                   brightness: float, contrast: float):
        """
        Property 23: Scale-Preserving Augmentation
        
        For any data augmentation applied during training, geometric scale 
        relationships needed for height measurement should be preserved.
        
        Validates Requirements 9.3, 9.4
        """
        # Generate synthetic wave image with geometric features
        image = self._create_synthetic_wave_image(height, width, brightness, contrast)
        
        # Apply augmentation
        result = self.augmentation.augment_training_sample(
            image, preserve_labels=True
        )
        
        # Verify augmentation succeeded
        assert result['augmentation_success'], "Augmentation should succeed"
        
        # Verify scale preservation
        assert result['scale_preserved'], "Scale relationships must be preserved"
        
        # Verify image dimensions are unchanged (critical for scale)
        original_shape = image.shape
        augmented_shape = result['augmented_image'].shape
        assert original_shape == augmented_shape, "Image dimensions must remain unchanged"
        
        # Verify geometric features are preserved
        scale_validation = self.augmentation.validate_scale_preservation(
            image, result['augmented_image']
        )
        assert scale_validation, "Geometric scale validation must pass"
        
        # Verify edge preservation for wave boundaries
        edge_preservation_ratio = self._calculate_edge_preservation(
            image, result['augmented_image']
        )
        assert edge_preservation_ratio >= self.config.edge_preservation_threshold, \
            f"Edge preservation ratio {edge_preservation_ratio:.3f} below threshold"
        
        # Verify structural similarity for height measurement accuracy
        structural_similarity = self._calculate_structural_similarity(
            image, result['augmented_image']
        )
        assert structural_similarity >= (1.0 - self.config.scale_tolerance), \
            f"Structural similarity {structural_similarity:.3f} indicates scale distortion"
    
    @given(
        weather_type=st.sampled_from(['fog', 'rain', 'glare']),
        intensity=st.floats(min_value=0.05, max_value=0.4)
    )
    @settings(max_examples=30, deadline=5000)
    def test_weather_effects_preserve_scale(self, weather_type: str, intensity: float):
        """Test that weather effects preserve geometric scale."""
        # Create test image with known geometric features
        image = self._create_geometric_test_pattern(512, 512)
        
        # Configure specific weather effect
        config = AugmentationConfig(
            enable_weather_effects=True,
            enable_lighting_changes=False,
            preserve_scale=True
        )
        
        # Set specific weather probabilities
        if weather_type == 'fog':
            config.fog_probability = 1.0
            config.fog_intensity_range = (intensity, intensity)
        elif weather_type == 'rain':
            config.rain_probability = 1.0
            config.rain_intensity_range = (intensity, intensity)
        elif weather_type == 'glare':
            config.glare_probability = 1.0
            config.glare_intensity_range = (intensity, intensity)
        
        augmentation = WaveAugmentation(config)
        
        # Apply weather effect
        augmented = augmentation.apply_weather_effects(image)
        
        # Verify scale preservation
        scale_preserved = augmentation.validate_scale_preservation(image, augmented)
        assert scale_preserved, f"{weather_type} effect should preserve scale"
        
        # Verify geometric measurements are consistent
        original_features = self._extract_geometric_features(image)
        augmented_features = self._extract_geometric_features(augmented)
        
        # Check that relative distances are preserved
        for i, (orig_feat, aug_feat) in enumerate(zip(original_features, augmented_features)):
            relative_error = abs(orig_feat - aug_feat) / max(orig_feat, 1e-6)
            assert relative_error <= self.config.scale_tolerance, \
                f"Feature {i} scale changed by {relative_error:.3f} > tolerance"
    
    @given(
        brightness_change=st.floats(min_value=-0.2, max_value=0.2),
        contrast_change=st.floats(min_value=-0.2, max_value=0.2)
    )
    @settings(max_examples=30, deadline=5000)
    def test_lighting_changes_preserve_scale(self, brightness_change: float, 
                                           contrast_change: float):
        """Test that lighting changes preserve geometric scale."""
        # Create test image
        image = self._create_geometric_test_pattern(512, 512)
        
        # Configure lighting changes
        config = AugmentationConfig(
            enable_weather_effects=False,
            enable_lighting_changes=True,
            preserve_scale=True,
            max_brightness_change=abs(brightness_change),
            max_contrast_change=abs(contrast_change)
        )
        
        augmentation = WaveAugmentation(config)
        
        # Apply lighting changes
        augmented = augmentation.apply_lighting_changes(image)
        
        # Verify scale preservation
        scale_preserved = augmentation.validate_scale_preservation(image, augmented)
        assert scale_preserved, "Lighting changes should preserve scale"
        
        # Verify edge structure is maintained
        original_edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
        augmented_edges = cv2.Canny(cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY), 50, 150)
        
        edge_correlation = cv2.matchTemplate(
            original_edges.astype(np.float32), 
            augmented_edges.astype(np.float32), 
            cv2.TM_CCOEFF_NORMED
        )
        max_correlation = np.max(edge_correlation)
        
        assert max_correlation >= 0.8, \
            f"Edge correlation {max_correlation:.3f} too low for lighting changes"
    
    def _create_synthetic_wave_image(self, height: int, width: int, 
                                   brightness: float, contrast: float) -> np.ndarray:
        """Create synthetic wave image with geometric features."""
        # Create base ocean surface
        image = np.full((height, width, 3), brightness, dtype=np.uint8)
        
        # Add wave patterns with known geometry
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 2 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Create wave pattern
        wave_pattern = np.sin(X) * np.cos(Y * 0.5) * contrast * 50
        
        # Apply to all channels
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            channel += wave_pattern
            image[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return image
    
    def _create_geometric_test_pattern(self, height: int, width: int) -> np.ndarray:
        """Create test pattern with known geometric features."""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add geometric shapes with known dimensions
        center_x, center_y = width // 2, height // 2
        
        # Draw circles with known radii
        cv2.circle(image, (center_x, center_y), 50, (255, 255, 255), 2)
        cv2.circle(image, (center_x, center_y), 100, (200, 200, 200), 2)
        
        # Draw rectangles with known dimensions
        cv2.rectangle(image, (center_x - 75, center_y - 25), 
                     (center_x + 75, center_y + 25), (150, 150, 150), 2)
        
        # Add horizontal lines for scale reference
        for y in range(center_y - 150, center_y + 150, 30):
            cv2.line(image, (center_x - 200, y), (center_x + 200, y), (100, 100, 100), 1)
        
        return image
    
    def _calculate_edge_preservation(self, original: np.ndarray, 
                                   augmented: np.ndarray) -> float:
        """Calculate edge preservation ratio."""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        aug_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
        
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        aug_edges = cv2.Canny(aug_gray, 50, 150)
        
        orig_edge_count = np.sum(orig_edges > 0)
        aug_edge_count = np.sum(aug_edges > 0)
        
        if orig_edge_count == 0:
            return 1.0
        
        return aug_edge_count / orig_edge_count
    
    def _calculate_structural_similarity(self, original: np.ndarray, 
                                       augmented: np.ndarray) -> float:
        """Calculate structural similarity using template matching."""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        aug_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
        
        correlation = cv2.matchTemplate(orig_gray, aug_gray, cv2.TM_CCOEFF_NORMED)
        return np.max(correlation)
    
    def _extract_geometric_features(self, image: np.ndarray) -> list:
        """Extract geometric features for scale comparison."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, 
                                        qualityLevel=0.01, minDistance=30)
        
        if corners is None or len(corners) < 2:
            return [0.0]  # No features detected
        
        # Calculate distances between corner pairs
        features = []
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                dist = np.linalg.norm(corners[i] - corners[j])
                features.append(dist)
        
        return features[:5]  # Return first 5 distances


if __name__ == "__main__":
    # Run property-based tests
    pytest.main([__file__, "-v", "--tb=short"])