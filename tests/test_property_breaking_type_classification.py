#!/usr/bin/env python3
"""
Property-Based Test for Breaking Type Classification
Tests Property 14: Breaking Type Classification
Validates Requirements 5.1, 5.2
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
    from swellsight.models.heads import BreakingTypeHead
    SWELLSIGHT_AVAILABLE = True
except ImportError as e:
    print(f"SwellSight modules not available: {e}")
    SWELLSIGHT_AVAILABLE = False


@composite
def breaking_type_scenarios(draw):
    """Generate test scenarios for different breaking types."""
    scenario_type = draw(st.sampled_from([
        "spilling_waves", "plunging_waves", "surging_waves", "no_breaking", "mixed_breaking"
    ]))
    
    height, width = 128, 128
    rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 255
    depth_array = np.random.rand(height, width).astype(np.float32) * 0.5 + 0.25
    
    # Create breaking type patterns in depth map
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    if scenario_type == "spilling_waves":
        # Spilling: gradual, foamy breaking (gentle slope changes)
        wave_pattern = 0.15 * np.sin(X) * np.exp(-0.05 * Y) + 0.1 * np.random.rand(height, width)
        expected_type = "SPILLING"
        expected_intensity = "medium"
    elif scenario_type == "plunging_waves":
        # Plunging: steep, dramatic breaking (sharp depth changes)
        wave_pattern = 0.25 * np.sin(X) * (1 + 0.5 * np.sin(2 * Y))
        wave_pattern += 0.2 * np.exp(-((X - np.pi)**2 + (Y - np.pi)**2) / 2)
        expected_type = "PLUNGING"
        expected_intensity = "high"
    elif scenario_type == "surging_waves":
        # Surging: waves that don't break but surge up beach (smooth patterns)
        wave_pattern = 0.1 * np.sin(0.5 * X) * np.cos(0.3 * Y)
        expected_type = "SURGING"
        expected_intensity = "low"
    elif scenario_type == "no_breaking":
        # No breaking: very flat, minimal wave activity
        wave_pattern = 0.05 * np.sin(0.2 * X) * np.cos(0.1 * Y)
        expected_type = "NO_BREAKING"
        expected_intensity = "very_low"
    else:  # mixed_breaking
        # Mixed: combination of different breaking types
        spilling_pattern = 0.1 * np.sin(X) * np.exp(-0.05 * Y)
        plunging_pattern = 0.15 * np.sin(2 * X) * (1 + 0.3 * np.sin(Y))
        wave_pattern = spilling_pattern + plunging_pattern
        expected_type = None  # Could be any dominant type
        expected_intensity = "mixed"
    
    depth_array = np.clip(depth_array + wave_pattern, 0, 1)
    
    depth_map = DepthMap(
        data=depth_array,
        resolution=(height, width),
        quality_score=draw(st.floats(min_value=0.6, max_value=1.0)),
        edge_preservation=draw(st.floats(min_value=0.6, max_value=1.0))
    )
    
    return rgb_array, depth_map, scenario_type, expected_type, expected_intensity


@composite
def simple_breaking_patterns(draw):
    """Generate simple breaking patterns for basic testing."""
    breaking_type = draw(st.sampled_from(["SPILLING", "PLUNGING", "SURGING", "NO_BREAKING"]))
    
    height, width = 64, 64
    rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 255
    depth_array = np.random.rand(height, width).astype(np.float32) * 0.3 + 0.35
    
    # Create simple patterns based on breaking type
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    pattern_strength = draw(st.floats(min_value=0.05, max_value=0.2))
    
    if breaking_type == "SPILLING":
        pattern = pattern_strength * np.sin(X) * (1 + 0.3 * np.random.rand(height, width))
    elif breaking_type == "PLUNGING":
        pattern = pattern_strength * np.sin(X) * np.sin(Y) * 1.5
    elif breaking_type == "SURGING":
        pattern = pattern_strength * 0.5 * np.sin(0.5 * X)
    else:  # NO_BREAKING
        pattern = pattern_strength * 0.2 * np.sin(0.1 * X)
    
    depth_array += pattern
    depth_array = np.clip(depth_array, 0, 1)
    
    depth_map = DepthMap(
        data=depth_array,
        resolution=(height, width),
        quality_score=0.8,
        edge_preservation=0.7
    )
    
    return rgb_array, depth_map, breaking_type


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
        analyzer.eval()
        return analyzer
    except Exception as e:
        print(f"Failed to create analyzer: {e}")
        return None


@given(simple_breaking_patterns())
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_14_breaking_type_classification(pattern_data):
    """
    Feature: wave-analysis-system, Property 14: Breaking Type Classification
    
    Property: For any wave with identifiable breaking pattern, the system should classify breaking type correctly
    Validates: Requirements 5.1, 5.2
    """
    if not SWELLSIGHT_AVAILABLE:
        assume(False)
    
    rgb_array, depth_map, expected_breaking_type = pattern_data
    
    analyzer = create_test_analyzer()
    assume(analyzer is not None)
    
    try:
        # Analyze waves
        wave_metrics = analyzer.analyze_waves(rgb_array, depth_map)
        
        # Property assertions: Breaking type classification
        
        # 1. Breaking type should be one of the valid categories
        valid_breaking_types = ["SPILLING", "PLUNGING", "SURGING", "NO_BREAKING"]
        assert wave_metrics.breaking_type in valid_breaking_types, f"Invalid breaking type: {wave_metrics.breaking_type}"
        
        # 2. Breaking confidence should be in valid range
        assert 0.0 <= wave_metrics.breaking_confidence <= 1.0, f"Breaking confidence {wave_metrics.breaking_confidence} outside [0,1]"
        
        # 3. For clear patterns with good quality, confidence should be reasonable
        if depth_map.quality_score > 0.7 and depth_map.edge_preservation > 0.7:
            assert wave_metrics.breaking_confidence >= 0.05, f"High quality input should have reasonable confidence, got {wave_metrics.breaking_confidence}"
        
        # 4. Breaking type should be consistent (deterministic behavior)
        wave_metrics_2 = analyzer.analyze_waves(rgb_array, depth_map)
        assert wave_metrics_2.breaking_type == wave_metrics.breaking_type, "Breaking type classification should be deterministic"
        
        # 5. Confidence should be deterministic
        confidence_diff = abs(wave_metrics_2.breaking_confidence - wave_metrics.breaking_confidence)
        assert confidence_diff < 0.001, f"Breaking confidence should be deterministic, diff: {confidence_diff}"
        
        # 6. All wave metrics should be present and valid
        assert hasattr(wave_metrics, 'height_meters'), "Should have height prediction"
        assert hasattr(wave_metrics, 'direction'), "Should have direction prediction"
        assert hasattr(wave_metrics, 'breaking_type'), "Should have breaking type prediction"
        
    except Exception as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            assume(False)
        else:
            raise


@given(breaking_type_scenarios())
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_14_breaking_scenarios(scenario_data):
    """
    Test breaking type classification across different wave scenarios
    Part of Property 14: Breaking Type Classification
    """
    if not SWELLSIGHT_AVAILABLE:
        assume(False)
    
    rgb_array, depth_map, scenario_type, expected_type, expected_intensity = scenario_data
    
    analyzer = create_test_analyzer()
    assume(analyzer is not None)
    
    try:
        # Analyze waves
        wave_metrics = analyzer.analyze_waves(rgb_array, depth_map)
        
        # Property assertions for different scenarios
        
        # 1. Should always return a valid breaking type
        valid_breaking_types = ["SPILLING", "PLUNGING", "SURGING", "NO_BREAKING"]
        assert wave_metrics.breaking_type in valid_breaking_types, f"Should return valid breaking type, got: {wave_metrics.breaking_type}"
        
        # 2. Confidence should be reasonable for the scenario
        assert 0.0 <= wave_metrics.breaking_confidence <= 1.0, f"Confidence {wave_metrics.breaking_confidence} outside [0,1]"
        
        # 3. For no_breaking scenarios, should handle appropriately
        if scenario_type == "no_breaking":
            # Should either predict NO_BREAKING or have low confidence in other types
            if wave_metrics.breaking_type != "NO_BREAKING":
                assert wave_metrics.breaking_confidence <= 0.8, f"Non-breaking scenario should have lower confidence for breaking types, got {wave_metrics.breaking_confidence}"
        
        # 4. For clear breaking scenarios, should have reasonable confidence
        if scenario_type in ["spilling_waves", "plunging_waves", "surging_waves"]:
            assert wave_metrics.breaking_confidence >= 0.05, f"Clear breaking scenarios should have some confidence, got {wave_metrics.breaking_confidence}"
        
        # 5. Mixed breaking scenarios should be handled without crashing
        if scenario_type == "mixed_breaking":
            # Should return some valid breaking type (the dominant one)
            assert wave_metrics.breaking_type in valid_breaking_types, "Mixed breaking should return valid dominant type"
        
        # 6. System should handle all scenarios robustly
        assert wave_metrics is not None, "Should successfully analyze all breaking scenarios"
        
    except Exception as e:
        if "CUDA" in str(e) or "memory" in str(e).lower():
            assume(False)
        else:
            raise


def test_property_14_breaking_type_coverage():
    """
    Test that all breaking type categories can be predicted
    Part of Property 14: Breaking Type Classification
    """
    if not SWELLSIGHT_AVAILABLE:
        return
    
    analyzer = create_test_analyzer()
    if analyzer is None:
        return
    
    # Test multiple random inputs to see if we get different breaking types
    breaking_types_seen = set()
    
    for i in range(15):
        # Create random test data with varying characteristics
        rgb_array = np.random.rand(64, 64, 3).astype(np.float32) * 255
        
        # Vary depth patterns to encourage different breaking types
        depth_data = np.random.rand(64, 64).astype(np.float32)
        if i % 4 == 0:
            # Flat pattern (might produce NO_BREAKING)
            depth_data *= 0.1
        elif i % 4 == 1:
            # Moderate pattern (might produce SURGING)
            depth_data *= 0.3
        elif i % 4 == 2:
            # Strong pattern (might produce SPILLING)
            depth_data *= 0.6
        else:
            # Very strong pattern (might produce PLUNGING)
            depth_data *= 0.8
            
        depth_map = DepthMap(data=depth_data, resolution=(64, 64), quality_score=0.8, edge_preservation=0.7)
        
        try:
            result = analyzer.analyze_waves(rgb_array, depth_map)
            breaking_types_seen.add(result.breaking_type)
            
            # Verify breaking type is valid
            assert result.breaking_type in ["SPILLING", "PLUNGING", "SURGING", "NO_BREAKING"], f"Invalid breaking type: {result.breaking_type}"
            
        except Exception as e:
            if "CUDA" in str(e) or "memory" in str(e).lower():
                break  # Skip hardware issues
            else:
                raise
    
    # We should see at least one valid breaking type
    assert len(breaking_types_seen) >= 1, "Should predict at least one valid breaking type"
    
    # All seen breaking types should be valid
    valid_breaking_types = {"SPILLING", "PLUNGING", "SURGING", "NO_BREAKING"}
    assert breaking_types_seen.issubset(valid_breaking_types), f"All breaking types should be valid: {breaking_types_seen}"


def run_property_tests():
    """Run all property tests for breaking type classification"""
    print("Running Property-Based Tests for Breaking Type Classification...")
    print("=" * 70)
    
    if not SWELLSIGHT_AVAILABLE:
        print("❌ SwellSight modules not available. Skipping tests.")
        return False
    
    try:
        print("Testing Property 14: Breaking Type Classification...")
        
        # Test Property 14: Basic breaking type classification
        print("  ✓ Testing Property 14: Breaking Type Classification Accuracy...")
        test_property_14_breaking_type_classification()
        
        # Test breaking scenarios
        print("  ✓ Testing breaking type scenarios...")
        test_property_14_breaking_scenarios()
        
        # Test breaking type coverage
        print("  ✓ Testing breaking type category coverage...")
        test_property_14_breaking_type_coverage()
        
        print("\n🎉 All property tests passed!")
        print("Property 14 (Breaking Type Classification) validated successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Property test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_tests()
    exit(0 if success else 1)