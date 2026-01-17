#!/usr/bin/env python3
"""
Property-Based Test for Multi-Task Input Processing
Tests Property 18: Multi-Task Input Processing
Validates Requirements 7.1, 7.2: Multi-task model should process 4-channel input and generate all three output types
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import tempfile
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    print("Hypothesis not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hypothesis"])
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True

try:
    from swellsight.core.wave_analyzer import DINOv2WaveAnalyzer, ConfidenceScores
    from swellsight.core.depth_extractor import DepthMap
    from swellsight.core.synthetic_generator import WaveMetrics
    SWELLSIGHT_AVAILABLE = True
except ImportError as e:
    print(f"SwellSight modules not available: {e}")
    SWELLSIGHT_AVAILABLE = False


@composite
def valid_4channel_input(draw):
    """Generate valid 4-channel (RGB + Depth) input tensors for testing."""
    # Generate smaller, reasonable image dimensions to avoid Hypothesis limits
    height = draw(st.integers(min_value=64, max_value=256))
    width = draw(st.integers(min_value=64, max_value=256))
    
    # Generate RGB array directly using numpy for efficiency
    rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 255.0
    
    # Generate depth array directly using numpy
    depth_array = np.random.rand(height, width).astype(np.float32)
    
    # Add some variation based on drawn parameters
    brightness = draw(st.floats(min_value=0.5, max_value=1.5))
    contrast = draw(st.floats(min_value=0.5, max_value=2.0))
    
    # Apply variations
    rgb_array = np.clip(rgb_array * brightness * contrast, 0, 255)
    depth_array = np.clip(depth_array * contrast, 0, 1)
    
    return rgb_array, depth_array, (height, width)


@composite
def depth_map_variations(draw):
    """Generate various DepthMap objects for testing."""
    height = draw(st.integers(min_value=64, max_value=256))
    width = draw(st.integers(min_value=64, max_value=256))
    
    # Generate normalized depth data using numpy for efficiency
    depth_array = np.random.rand(height, width).astype(np.float32)
    
    # Generate quality metrics
    quality_score = draw(st.floats(min_value=0.1, max_value=1.0))
    edge_preservation = draw(st.floats(min_value=0.1, max_value=1.0))
    
    return DepthMap(
        data=depth_array,
        resolution=(height, width),
        quality_score=quality_score,
        edge_preservation=edge_preservation
    )


def create_test_analyzer():
    """Create a DINOv2WaveAnalyzer instance for testing."""
    if not SWELLSIGHT_AVAILABLE:
        return None
    
    # Suppress logging during tests
    logging.getLogger().setLevel(logging.ERROR)
    
    try:
        analyzer = DINOv2WaveAnalyzer(
            backbone_model="dinov2_vitb14",
            freeze_backbone=True
        )
        analyzer.eval()  # Set to evaluation mode
        return analyzer
    except Exception as e:
        print(f"Failed to create analyzer: {e}")
        return None


@given(valid_4channel_input(), depth_map_variations())
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_18_multi_task_input_processing(input_data, depth_map):
    """
    Feature: wave-analysis-system, Property 18: Multi-Task Input Processing
    
    Property: For any 4-channel (RGB + Depth) input image, the Multi_Task_Model should process it correctly and generate all three output types
    Validates: Requirements 7.1, 7.2
    """
    if not SWELLSIGHT_AVAILABLE:
        assume(False)  # Skip test if modules not available
    
    rgb_array, depth_array, (height, width) = input_data
    
    # Ensure depth map matches input dimensions
    if depth_map.data.shape != (height, width):
        # Resize depth map to match input dimensions
        import cv2
        resized_depth = cv2.resize(depth_map.data, (width, height))
        depth_map = DepthMap(
            data=resized_depth,
            resolution=(height, width),
            quality_score=depth_map.quality_score,
            edge_preservation=depth_map.edge_preservation
        )
    
    # Create analyzer
    analyzer = create_test_analyzer()
    assume(analyzer is not None)
    
    try:
        # Test multi-task input processing
        wave_metrics = analyzer.analyze_waves(rgb_array, depth_map)
        
        # Property assertions: Model should process 4-channel input and generate all three output types
        
        # 1. Wave metrics should be returned
        assert wave_metrics is not None, "Wave metrics should not be None"
        assert isinstance(wave_metrics, WaveMetrics), "Should return WaveMetrics object"
        
        # 2. All three prediction types should be present
        # Height prediction (regression)
        assert hasattr(wave_metrics, 'height_meters'), "Should have height_meters prediction"
        assert hasattr(wave_metrics, 'height_feet'), "Should have height_feet prediction"
        assert hasattr(wave_metrics, 'height_confidence'), "Should have height confidence"
        
        assert isinstance(wave_metrics.height_meters, (int, float)), "Height meters should be numeric"
        assert isinstance(wave_metrics.height_feet, (int, float)), "Height feet should be numeric"
        assert isinstance(wave_metrics.height_confidence, (int, float)), "Height confidence should be numeric"
        
        # Direction prediction (classification)
        assert hasattr(wave_metrics, 'direction'), "Should have direction prediction"
        assert hasattr(wave_metrics, 'direction_confidence'), "Should have direction confidence"
        
        assert isinstance(wave_metrics.direction, str), "Direction should be string"
        assert wave_metrics.direction in ["LEFT", "RIGHT", "STRAIGHT"], f"Direction should be valid category, got {wave_metrics.direction}"
        assert isinstance(wave_metrics.direction_confidence, (int, float)), "Direction confidence should be numeric"
        
        # Breaking type prediction (classification)
        assert hasattr(wave_metrics, 'breaking_type'), "Should have breaking_type prediction"
        assert hasattr(wave_metrics, 'breaking_confidence'), "Should have breaking confidence"
        
        assert isinstance(wave_metrics.breaking_type, str), "Breaking type should be string"
        assert wave_metrics.breaking_type in ["SPILLING", "PLUNGING", "SURGING"], f"Breaking type should be valid category, got {wave_metrics.breaking_type}"
        assert isinstance(wave_metrics.breaking_confidence, (int, float)), "Breaking confidence should be numeric"
        
        # 3. Confidence scores should be in valid range [0, 1]
        confidence_scores = [
            wave_metrics.height_confidence,
            wave_metrics.direction_confidence,
            wave_metrics.breaking_confidence
        ]
        
        for i, conf in enumerate(confidence_scores):
            assert 0.0 <= conf <= 1.0, f"Confidence score {i} should be in [0,1], got {conf}"
        
        # 4. Height values should be in reasonable range
        assert 0.1 <= wave_metrics.height_meters <= 15.0, f"Height meters should be reasonable, got {wave_metrics.height_meters}"
        assert 0.3 <= wave_metrics.height_feet <= 50.0, f"Height feet should be reasonable, got {wave_metrics.height_feet}"
        
        # 5. Height conversion should be accurate (meters to feet)
        expected_feet = wave_metrics.height_meters * 3.28084
        feet_error = abs(wave_metrics.height_feet - expected_feet)
        assert feet_error < 0.01, f"Height conversion error too large: {feet_error}"
        
        # 6. Extreme conditions flag should be consistent
        assert isinstance(wave_metrics.extreme_conditions, bool), "Extreme conditions should be boolean"
        if wave_metrics.height_meters < 0.5 or wave_metrics.height_meters > 8.0:
            assert wave_metrics.extreme_conditions, "Should flag extreme conditions for unusual heights"
        
        # 7. Confidence scores should be accessible separately
        confidence_obj = analyzer.get_confidence_scores()
        assert isinstance(confidence_obj, ConfidenceScores), "Should return ConfidenceScores object"
        
        assert hasattr(confidence_obj, 'height_confidence'), "ConfidenceScores should have height_confidence"
        assert hasattr(confidence_obj, 'direction_confidence'), "ConfidenceScores should have direction_confidence"
        assert hasattr(confidence_obj, 'breaking_type_confidence'), "ConfidenceScores should have breaking_type_confidence"
        assert hasattr(confidence_obj, 'overall_confidence'), "ConfidenceScores should have overall_confidence"
        
        # Overall confidence should be reasonable average
        expected_overall = (confidence_obj.height_confidence + confidence_obj.direction_confidence + confidence_obj.breaking_type_confidence) / 3.0
        overall_error = abs(confidence_obj.overall_confidence - expected_overall)
        assert overall_error < 0.01, f"Overall confidence calculation error: {overall_error}"
        
    except Exception as e:
        # Log the error for debugging but don't fail the test for infrastructure issues
        if "CUDA" in str(e) or "memory" in str(e).lower() or "device" in str(e).lower():
            assume(False)  # Skip test for hardware-related issues
        else:
            raise  # Re-raise for actual logic errors


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_18_batch_processing_consistency(batch_size):
    """
    Test that multi-task processing is consistent across different batch sizes
    Part of Property 18: Multi-Task Input Processing
    """
    if not SWELLSIGHT_AVAILABLE:
        assume(False)
    
    analyzer = create_test_analyzer()
    assume(analyzer is not None)
    
    # Create test data
    height, width = 518, 518  # Standard input size
    rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 255
    depth_array = np.random.rand(height, width).astype(np.float32)
    depth_map = DepthMap(
        data=depth_array,
        resolution=(height, width),
        quality_score=0.8,
        edge_preservation=0.7
    )
    
    try:
        # Process single image multiple times
        results = []
        for _ in range(batch_size):
            wave_metrics = analyzer.analyze_waves(rgb_array, depth_map)
            results.append(wave_metrics)
        
        # All results should be identical (deterministic processing)
        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert abs(result.height_meters - base_result.height_meters) < 0.001, f"Height inconsistency at batch {i}"
            assert result.direction == base_result.direction, f"Direction inconsistency at batch {i}"
            assert result.breaking_type == base_result.breaking_type, f"Breaking type inconsistency at batch {i}"
            
    except Exception as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            assume(False)
        else:
            raise


def test_property_18_invalid_input_handling():
    """
    Test that the model handles invalid inputs gracefully
    Part of Property 18: Multi-Task Input Processing
    """
    if not SWELLSIGHT_AVAILABLE:
        return  # Skip if modules not available
    
    analyzer = create_test_analyzer()
    if analyzer is None:
        return
    
    # Test with invalid RGB shape (wrong number of channels)
    try:
        invalid_rgb = np.random.rand(256, 256, 2).astype(np.float32)  # Only 2 channels
        depth_array = np.random.rand(256, 256).astype(np.float32)
        depth_map = DepthMap(data=depth_array, resolution=(256, 256), quality_score=0.8, edge_preservation=0.7)
        
        # Should handle gracefully or raise informative error
        try:
            result = analyzer.analyze_waves(invalid_rgb, depth_map)
            # If it succeeds, result should still be valid
            assert isinstance(result, WaveMetrics), "Should return valid WaveMetrics even with unusual input"
        except (ValueError, RuntimeError) as e:
            # Acceptable to raise informative error
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ["channel", "shape", "size", "dimension"]), \
                f"Error should be informative about input issues, got: {str(e)}"
            
    except Exception as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            pass  # Skip hardware issues
        else:
            raise


def run_property_tests():
    """Run all property tests for multi-task input processing"""
    print("Running Property-Based Tests for Multi-Task Input Processing...")
    print("=" * 70)
    
    if not SWELLSIGHT_AVAILABLE:
        print("❌ SwellSight modules not available. Skipping tests.")
        return False
    
    try:
        print("Testing Property 18: Multi-Task Input Processing...")
        
        # Test 1: Basic multi-task processing
        print("  ✓ Testing 4-channel input processing with all output types...")
        test_property_18_multi_task_input_processing()
        
        # Test 2: Batch processing consistency
        print("  ✓ Testing batch processing consistency...")
        test_property_18_batch_processing_consistency()
        
        # Test 3: Invalid input handling
        print("  ✓ Testing invalid input handling...")
        test_property_18_invalid_input_handling()
        
        print("\n🎉 All property tests passed!")
        print("Property 18 (Multi-Task Input Processing) validated successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Property test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_tests()
    exit(0 if success else 1)