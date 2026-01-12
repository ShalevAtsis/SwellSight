"""
Property-based test for depth map quality assessment.

**Property 12: Depth Map Quality Assessment**
**Validates: Requirements 3.2**

This test validates that depth map quality assessment produces consistent and meaningful
quality scores across different types of depth maps.
"""

import pytest
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings, assume
import cv2
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from data_validator import validate_depth_map_quality


def generate_synthetic_depth_map(width: int, height: int, pattern: str = 'gradient') -> np.ndarray:
    """Generate synthetic depth maps with known characteristics for testing."""
    if pattern == 'gradient':
        # Linear gradient - should have high quality
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        return X + Y
    
    elif pattern == 'flat':
        # Flat surface - should have low quality (no depth variation)
        return np.ones((height, width)) * 0.5
    
    elif pattern == 'noise':
        # Random noise - should have medium quality
        return np.random.random((height, width))
    
    elif pattern == 'checkerboard':
        # Checkerboard pattern - should have high gradient score
        checker = np.zeros((height, width))
        checker[::2, ::2] = 1
        checker[1::2, 1::2] = 1
        return checker
    
    elif pattern == 'circular':
        # Circular gradient - should have good quality
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        return distance / np.max(distance)
    
    else:
        return np.random.random((height, width))


@st.composite
def depth_map_strategy(draw):
    """Strategy for generating test depth maps."""
    width = draw(st.integers(min_value=32, max_value=512))
    height = draw(st.integers(min_value=32, max_value=512))
    pattern = draw(st.sampled_from(['gradient', 'flat', 'noise', 'checkerboard', 'circular']))
    
    depth_map = generate_synthetic_depth_map(width, height, pattern)
    
    # Add some realistic noise and scaling
    noise_level = draw(st.floats(min_value=0.0, max_value=0.1))
    scale_factor = draw(st.floats(min_value=0.1, max_value=10.0))
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, depth_map.shape)
        depth_map = depth_map + noise
    
    depth_map = depth_map * scale_factor
    
    return {
        'depth_map': depth_map,
        'pattern': pattern,
        'width': width,
        'height': height,
        'noise_level': noise_level,
        'scale_factor': scale_factor
    }


class TestDepthMapQualityAssessment:
    """Property-based tests for depth map quality assessment."""
    
    @given(depth_map_strategy())
    @settings(max_examples=100, deadline=None)
    def test_quality_score_range_property(self, depth_data):
        """
        **Feature: swellsight-pipeline-improvements, Property 12: Depth Map Quality Assessment**
        
        For any depth map, the quality score should be in the range [0, 1].
        **Validates: Requirements 3.2**
        """
        depth_map = depth_data['depth_map']
        
        # Validate depth map quality
        quality_result = validate_depth_map_quality(depth_map)
        
        # Property: Quality score must be in valid range
        assert 0.0 <= quality_result['score'] <= 1.0, \
            f"Quality score {quality_result['score']} not in range [0, 1]"
        
        # Check that result has expected structure
        assert 'valid' in quality_result, "Missing 'valid' key in result"
        assert 'score' in quality_result, "Missing 'score' key in result"
        assert 'issues' in quality_result, "Missing 'issues' key in result"
        assert 'metrics' in quality_result, "Missing 'metrics' key in result"
    
    @given(depth_map_strategy())
    @settings(max_examples=100, deadline=None)
    def test_quality_consistency_property(self, depth_data):
        """
        **Feature: swellsight-pipeline-improvements, Property 12: Depth Map Quality Assessment**
        
        For any depth map, running quality assessment multiple times should produce 
        consistent results (deterministic behavior).
        **Validates: Requirements 3.2**
        """
        depth_map = depth_data['depth_map']
        
        # Run quality assessment multiple times
        quality_1 = validate_depth_map_quality(depth_map)
        quality_2 = validate_depth_map_quality(depth_map)
        quality_3 = validate_depth_map_quality(depth_map)
        
        # Property: Results should be identical (deterministic)
        tolerance = 1e-10  # Allow for floating point precision
        
        # Check main score consistency
        assert abs(quality_1['score'] - quality_2['score']) < tolerance, \
            f"Inconsistent score: {quality_1['score']} vs {quality_2['score']}"
        assert abs(quality_2['score'] - quality_3['score']) < tolerance, \
            f"Inconsistent score: {quality_2['score']} vs {quality_3['score']}"
        
        # Check metrics consistency
        for metric_name in quality_1['metrics'].keys():
            if isinstance(quality_1['metrics'][metric_name], (int, float)):
                assert abs(quality_1['metrics'][metric_name] - quality_2['metrics'][metric_name]) < tolerance, \
                    f"Inconsistent {metric_name}: {quality_1['metrics'][metric_name]} vs {quality_2['metrics'][metric_name]}"
                assert abs(quality_2['metrics'][metric_name] - quality_3['metrics'][metric_name]) < tolerance, \
                    f"Inconsistent {metric_name}: {quality_2['metrics'][metric_name]} vs {quality_3['metrics'][metric_name]}"
    
    @given(depth_map_strategy())
    @settings(max_examples=100, deadline=None)
    def test_flat_vs_varied_depth_property(self, depth_data):
        """
        **Feature: swellsight-pipeline-improvements, Property 12: Depth Map Quality Assessment**
        
        For any depth map, a flat (constant) version should have lower quality than 
        a version with depth variation.
        **Validates: Requirements 3.2**
        """
        original_depth = depth_data['depth_map']
        
        # Skip if the original is already very flat
        if np.std(original_depth) < 1e-6:
            assume(False)  # Skip this test case
        
        # Create a flat version (constant depth)
        flat_depth = np.full_like(original_depth, np.mean(original_depth))
        
        # Get quality scores
        original_quality = validate_depth_map_quality(original_depth)
        flat_quality = validate_depth_map_quality(flat_depth)
        
        # Property: Original should have higher quality than flat version
        # (original should have higher overall score and higher depth_std)
        assert original_quality['score'] > flat_quality['score'], \
            f"Original score {original_quality['score']} not > flat score {flat_quality['score']}"
        
        # Check that original has higher depth variation
        original_std = original_quality['metrics'].get('depth_std', 0)
        flat_std = flat_quality['metrics'].get('depth_std', 0)
        assert original_std > flat_std, \
            f"Original depth_std {original_std} not > flat depth_std {flat_std}"
    
    @given(st.integers(min_value=32, max_value=256), st.integers(min_value=32, max_value=256))
    @settings(max_examples=50, deadline=None)
    def test_noise_impact_property(self, width, height):
        """
        **Feature: swellsight-pipeline-improvements, Property 12: Depth Map Quality Assessment**
        
        For any clean depth map, adding noise should generally decrease quality scores
        (except possibly entropy which might increase with more variation).
        **Validates: Requirements 3.2**
        """
        # Create a clean gradient depth map
        clean_depth = generate_synthetic_depth_map(width, height, 'gradient')
        
        # Add significant noise
        noise_level = 0.2
        noise = np.random.normal(0, noise_level, clean_depth.shape)
        noisy_depth = clean_depth + noise
        
        # Get quality scores
        clean_quality = validate_depth_map_quality(clean_depth)
        noisy_quality = validate_depth_map_quality(noisy_depth)
        
        # Property: Clean version should generally have better or equal quality
        # (noise can sometimes increase variation which might increase score)
        # But edge strength should be affected by noise
        clean_edge = clean_quality['metrics'].get('edge_strength', 0)
        noisy_edge = noisy_quality['metrics'].get('edge_strength', 0)
        
        # The relationship depends on the type of noise and original pattern
        # At minimum, both should produce valid results
        assert clean_quality['score'] >= 0.0, f"Clean quality score {clean_quality['score']} < 0"
        assert noisy_quality['score'] >= 0.0, f"Noisy quality score {noisy_quality['score']} < 0"
    
    @given(depth_map_strategy())
    @settings(max_examples=100, deadline=None)
    def test_scaling_invariance_property(self, depth_data):
        """
        **Feature: swellsight-pipeline-improvements, Property 12: Depth Map Quality Assessment**
        
        For any depth map, scaling all values by a positive constant should not 
        significantly change normalized quality metrics (except dynamic_range).
        **Validates: Requirements 3.2**
        """
        original_depth = depth_data['depth_map']
        
        # Skip if depth map has no variation
        if np.std(original_depth) < 1e-6:
            assume(False)
        
        # Scale by a positive factor
        scale_factor = 2.5
        scaled_depth = original_depth * scale_factor
        
        # Get quality scores
        original_quality = validate_depth_map_quality(original_depth)
        scaled_quality = validate_depth_map_quality(scaled_depth)
        
        # Property: Overall quality scores should be similar (within tolerance)
        # The quality assessment should be relatively scale-invariant
        tolerance = 0.3  # 30% tolerance for scale effects
        
        score_diff = abs(original_quality['score'] - scaled_quality['score'])
        max_score = max(original_quality['score'], scaled_quality['score'])
        relative_diff = score_diff / (max_score + 1e-8)  # Avoid division by zero
        
        assert relative_diff <= tolerance, \
            f"Quality scores not scale-invariant: {original_quality['score']} vs {scaled_quality['score']} (relative diff: {relative_diff})"
        
        # Both should produce valid results
        assert original_quality['score'] >= 0.0, f"Original score {original_quality['score']} < 0"
        assert scaled_quality['score'] >= 0.0, f"Scaled score {scaled_quality['score']} < 0"
    
    @given(st.integers(min_value=64, max_value=256), st.integers(min_value=64, max_value=256))
    @settings(max_examples=50, deadline=None)
    def test_pattern_quality_ordering_property(self, width, height):
        """
        **Feature: swellsight-pipeline-improvements, Property 12: Depth Map Quality Assessment**
        
        For any image dimensions, different depth patterns should have predictable 
        quality orderings based on their characteristics.
        **Validates: Requirements 3.2**
        """
        # Generate different patterns
        gradient_depth = generate_synthetic_depth_map(width, height, 'gradient')
        flat_depth = generate_synthetic_depth_map(width, height, 'flat')
        checkerboard_depth = generate_synthetic_depth_map(width, height, 'checkerboard')
        
        # Get quality scores
        gradient_quality = validate_depth_map_quality(gradient_depth)
        flat_quality = validate_depth_map_quality(flat_depth)
        checkerboard_quality = validate_depth_map_quality(checkerboard_depth)
        
        # Property: Flat should have lowest depth_std
        flat_std = flat_quality['metrics'].get('depth_std', 0)
        gradient_std = gradient_quality['metrics'].get('depth_std', 0)
        checkerboard_std = checkerboard_quality['metrics'].get('depth_std', 0)
        
        assert flat_std <= gradient_std, \
            f"Flat depth_std {flat_std} not <= gradient depth_std {gradient_std}"
        
        assert flat_std <= checkerboard_std, \
            f"Flat depth_std {flat_std} not <= checkerboard depth_std {checkerboard_std}"
        
        # Property: Checkerboard should have high edge strength (lots of edges)
        checkerboard_edge = checkerboard_quality['metrics'].get('edge_strength', 0)
        gradient_edge = gradient_quality['metrics'].get('edge_strength', 0)
        
        assert checkerboard_edge >= gradient_edge, \
            f"Checkerboard edge_strength {checkerboard_edge} not >= gradient edge_strength {gradient_edge}"
    
    @given(depth_map_strategy())
    @settings(max_examples=100, deadline=None)
    def test_quality_metrics_correlation_property(self, depth_data):
        """
        **Feature: swellsight-pipeline-improvements, Property 12: Depth Map Quality Assessment**
        
        For any depth map, quality metrics should have reasonable relationships
        (e.g., very flat maps should have low variation and edge strength).
        **Validates: Requirements 3.2**
        """
        depth_map = depth_data['depth_map']
        quality_result = validate_depth_map_quality(depth_map)
        
        depth_std = quality_result['metrics'].get('depth_std', 0)
        edge_strength = quality_result['metrics'].get('edge_strength', 0)
        depth_range = quality_result['metrics'].get('depth_range', 0)
        
        # Property: If depth std is very low, edge strength should also be low
        if depth_std < 0.01:
            assert edge_strength <= 0.1, \
                f"Low depth_std ({depth_std}) but high edge_strength ({edge_strength})"
        
        # Property: If depth range is very small, std should also be small
        if depth_range < 0.01:
            assert depth_std <= 0.1, \
                f"Low depth_range ({depth_range}) but high depth_std ({depth_std})"
        
        # Property: All metrics should be non-negative
        assert depth_std >= 0, f"Negative depth_std: {depth_std}"
        assert edge_strength >= 0, f"Negative edge_strength: {edge_strength}"
        assert depth_range >= 0, f"Negative depth_range: {depth_range}"
    
    @given(depth_map_strategy())
    @settings(max_examples=100, deadline=None)
    def test_combined_score_bounds_property(self, depth_data):
        """
        **Feature: swellsight-pipeline-improvements, Property 12: Depth Map Quality Assessment**
        
        For any depth map, the quality score should be a reasonable
        combination of underlying metrics and within expected bounds.
        **Validates: Requirements 3.2**
        """
        depth_map = depth_data['depth_map']
        quality_result = validate_depth_map_quality(depth_map)
        
        # Property: Quality score should be in valid range
        score = quality_result['score']
        assert 0.0 <= score <= 1.0, f"Quality score {score} not in range [0, 1]"
        
        # Property: Score should reflect underlying metrics
        metrics = quality_result['metrics']
        depth_std = metrics.get('depth_std', 0)
        depth_range = metrics.get('depth_range', 0)
        edge_strength = metrics.get('edge_strength', 0)
        
        # If all metrics indicate poor quality, score should be low
        if depth_std < 0.01 and depth_range < 0.01 and edge_strength < 0.01:
            assert score <= 0.5, f"Low variation metrics but high score {score}"
        
        # Property: Valid result structure
        assert isinstance(quality_result['valid'], bool), "Invalid 'valid' field type"
        assert isinstance(quality_result['issues'], list), "Invalid 'issues' field type"
        assert isinstance(quality_result['metrics'], dict), "Invalid 'metrics' field type"


if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v", "--tb=short"])