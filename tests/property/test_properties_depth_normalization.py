"""
Property-based tests for depth map normalization.

Tests Property 7: Depth Map Normalization
Validates: Requirements 2.4
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from swellsight.core.depth_extractor import DepthAnythingV2Extractor, DepthMap

def generate_test_depth_map(width: int, height: int, wave_pattern: str) -> DepthMap:
    """Generate synthetic depth map for testing normalization."""
    depth_data = np.zeros((height, width), dtype=np.float32)
    
    if wave_pattern == "flat":
        # Flat ocean surface with minimal variation
        depth_data.fill(0.5)
        depth_data += np.random.normal(0, 0.01, (height, width))
        
    elif wave_pattern == "waves":
        # Sinusoidal wave pattern
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 2 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        depth_data = 0.5 + 0.2 * np.sin(X) * np.cos(Y)
        
    elif wave_pattern == "gradient":
        # Depth gradient from near to far
        for y in range(height):
            depth_data[y, :] = y / height
            
    elif wave_pattern == "random":
        # Random depth values
        depth_data = np.random.random((height, width))
    
    # Normalize to [0, 1]
    depth_data = np.clip(depth_data, 0, 1)
    
    # Calculate basic quality metrics
    quality_score = np.std(depth_data)
    edge_preservation = 0.5  # Default value
    
    return DepthMap(
        data=depth_data,
        resolution=(width, height),
        quality_score=quality_score,
        edge_preservation=edge_preservation
    )

# Hypothesis strategies
depth_map_strategy = st.builds(
    generate_test_depth_map,
    width=st.integers(min_value=320, max_value=640),
    height=st.integers(min_value=240, max_value=480),
    wave_pattern=st.sampled_from(["flat", "waves", "gradient", "random"])
)

enhancement_factor_strategy = st.floats(min_value=0.5, max_value=3.0)

class TestDepthMapNormalization:
    """Property-based tests for depth map normalization."""
    
    @given(
        depth_map=depth_map_strategy,
        enhancement_factor=enhancement_factor_strategy
    )
    @settings(max_examples=3, suppress_health_check=[HealthCheck.too_slow], deadline=60000)
    def test_property_7_depth_map_normalization(self, depth_map, enhancement_factor):
        """
        Property 7: Depth Map Normalization
        
        For any generated depth map, normalization should ensure waves stand out 
        against the ocean surface with measurable contrast ratios.
        
        **Feature: wave-analysis-system, Property 7: Depth Map Normalization**
        **Validates: Requirements 2.4**
        """
        # Initialize extractor
        extractor = DepthAnythingV2Extractor(model_size="small", precision="fp32")
        
        try:
            # Apply normalization
            normalized_depth_map = extractor.normalize_depth_for_waves(depth_map, enhancement_factor)
            
            # Validate basic properties
            assert isinstance(normalized_depth_map, DepthMap), "Should return DepthMap object"
            assert normalized_depth_map.data.shape == depth_map.data.shape, "Shape should be preserved"
            assert normalized_depth_map.resolution == depth_map.resolution, "Resolution should be preserved"
            
            # Validate depth values are still in [0, 1] range
            assert 0 <= normalized_depth_map.data.min() <= normalized_depth_map.data.max() <= 1, \
                f"Normalized depth values should be in [0,1], got [{normalized_depth_map.data.min():.3f}, {normalized_depth_map.data.max():.3f}]"
            
            # Test contrast enhancement
            original_contrast = np.std(depth_map.data)
            normalized_contrast = np.std(normalized_depth_map.data)
            
            # Normalization should generally maintain or improve contrast
            # (unless the original was already very high contrast)
            if original_contrast > 0.01:  # Only test if original had some variation
                contrast_ratio = normalized_contrast / original_contrast
                assert contrast_ratio >= 0.4, f"Normalization should not drastically reduce contrast, got ratio {contrast_ratio:.3f}"
            
            # Test that normalization preserves overall structure
            # Calculate correlation between original and normalized
            if np.std(depth_map.data) > 0.001 and np.std(normalized_depth_map.data) > 0.001:
                correlation = np.corrcoef(depth_map.data.flatten(), normalized_depth_map.data.flatten())[0, 1]
                assert correlation >= 0.2, f"Normalization should preserve structure, got correlation {correlation:.3f}"
            
            # Test quality metrics improvement or maintenance
            original_quality = depth_map.quality_score
            normalized_quality = normalized_depth_map.quality_score
            
            # Quality should not become negative
            assert normalized_quality >= 0, f"Normalized quality should be non-negative, got {normalized_quality:.3f}"
            
        except Exception as e:
            pytest.fail(f"Depth normalization failed: {e}")
    
    @given(depth_map=depth_map_strategy)
    @settings(max_examples=2, suppress_health_check=[HealthCheck.too_slow], deadline=60000)
    def test_normalization_idempotency(self, depth_map):
        """
        Test that applying normalization multiple times doesn't drastically change results.
        
        **Feature: wave-analysis-system, Property 7: Depth Map Normalization**
        **Validates: Requirements 2.4**
        """
        extractor = DepthAnythingV2Extractor(model_size="small", precision="fp32")
        
        try:
            # Apply normalization twice
            normalized_once = extractor.normalize_depth_for_waves(depth_map, enhancement_factor=1.5)
            normalized_twice = extractor.normalize_depth_for_waves(normalized_once, enhancement_factor=1.5)
            
            # Results should be similar (not drastically different)
            if np.std(normalized_once.data) > 0.001 and np.std(normalized_twice.data) > 0.001:
                correlation = np.corrcoef(normalized_once.data.flatten(), normalized_twice.data.flatten())[0, 1]
                assert correlation >= 0.7, f"Double normalization should be similar to single, got correlation {correlation:.3f}"
            
            # Both should still be in valid range
            assert 0 <= normalized_twice.data.min() <= normalized_twice.data.max() <= 1, \
                "Double normalized depth should still be in [0,1] range"
                
        except Exception as e:
            pytest.fail(f"Normalization idempotency test failed: {e}")
    
    @given(
        width=st.integers(min_value=320, max_value=640),
        height=st.integers(min_value=240, max_value=480)
    )
    @settings(max_examples=2, suppress_health_check=[HealthCheck.too_slow], deadline=60000)
    def test_normalization_edge_cases(self, width, height):
        """
        Test normalization with edge case depth maps.
        
        **Feature: wave-analysis-system, Property 7: Depth Map Normalization**
        **Validates: Requirements 2.4**
        """
        extractor = DepthAnythingV2Extractor(model_size="small", precision="fp32")
        
        # Test edge cases
        edge_cases = [
            ("all_zeros", np.zeros((height, width), dtype=np.float32)),
            ("all_ones", np.ones((height, width), dtype=np.float32)),
            ("all_half", np.full((height, width), 0.5, dtype=np.float32)),
            ("binary", np.random.choice([0.0, 1.0], size=(height, width)).astype(np.float32))
        ]
        
        for case_name, depth_data in edge_cases:
            try:
                # Create depth map
                depth_map = DepthMap(
                    data=depth_data,
                    resolution=(width, height),
                    quality_score=0.5,
                    edge_preservation=0.5
                )
                
                # Apply normalization
                normalized = extractor.normalize_depth_for_waves(depth_map, enhancement_factor=2.0)
                
                # Should not crash and should return valid depth map
                assert isinstance(normalized, DepthMap), f"Should return DepthMap for {case_name}"
                assert normalized.data.shape == (height, width), f"Shape should be preserved for {case_name}"
                assert 0 <= normalized.data.min() <= normalized.data.max() <= 1, \
                    f"Values should be in [0,1] for {case_name}"
                
            except Exception as e:
                pytest.fail(f"Normalization failed for edge case {case_name}: {e}")
    
    def test_normalization_with_real_extractor_output(self):
        """
        Test normalization with actual depth extractor output.
        
        **Feature: wave-analysis-system, Property 7: Depth Map Normalization**
        **Validates: Requirements 2.4**
        """
        extractor = DepthAnythingV2Extractor(model_size="small", precision="fp32")
        
        # Create a realistic test image
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        try:
            # Extract depth map
            depth_map = extractor.extract_depth(test_image)
            
            # Apply normalization
            normalized = extractor.normalize_depth_for_waves(depth_map, enhancement_factor=1.8)
            
            # Validate results
            assert isinstance(normalized, DepthMap), "Should return DepthMap"
            assert normalized.data.shape == depth_map.data.shape, "Shape should be preserved"
            assert 0 <= normalized.data.min() <= normalized.data.max() <= 1, "Values should be in [0,1]"
            
            # Quality metrics should be reasonable
            assert 0 <= normalized.quality_score <= 1, "Quality score should be in [0,1]"
            assert normalized.edge_preservation >= -1, "Edge preservation should be reasonable"
            
        except Exception as e:
            pytest.fail(f"Real extractor normalization test failed: {e}")

if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v"])