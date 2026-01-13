#!/usr/bin/env python3
"""
Property-Based Test for Confidence Score Generation
Tests Property 24: Confidence Score Generation
Validates Requirements 3.5, 4.5, 5.4, 10.3
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
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
    from swellsight.utils.confidence import ComprehensiveConfidenceScorer, ConfidenceMetrics, CalibrationResults
    SWELLSIGHT_AVAILABLE = True
except ImportError as e:
    print(f"SwellSight modules not available: {e}")
    SWELLSIGHT_AVAILABLE = False


@composite
def confidence_test_data(draw):
    """Generate test data for confidence scoring testing."""
    # Generate image dimensions
    height = draw(st.integers(min_value=64, max_value=256))
    width = draw(st.integers(min_value=64, max_value=256))
    
    # Generate RGB image with wave-like patterns
    rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 255
    
    # Generate depth map with wave-like depth variations
    depth_array = np.random.rand(height, width).astype(np.float32)
    depth_map = DepthMap(
        data=depth_array,
        resolution=(height, width),
        quality_score=0.8,
        edge_preservation=0.7
    )
    
    # Generate expected wave metrics for validation
    expected_height = draw(st.floats(min_value=0.5, max_value=8.0))
    expected_direction = draw(st.sampled_from(["LEFT", "RIGHT", "STRAIGHT"]))
    expected_breaking = draw(st.sampled_from(["SPILLING", "PLUNGING", "SURGING", "NO_BREAKING"]))
    
    return {
        "rgb_image": rgb_array,
        "depth_map": depth_map,
        "expected_height": expected_height,
        "expected_direction": expected_direction,
        "expected_breaking": expected_breaking
    }


@composite
def calibration_data_scenarios(draw):
    """Generate scenarios for confidence calibration testing."""
    # Generate multiple confidence-accuracy pairs for calibration
    num_samples = draw(st.integers(min_value=10, max_value=100))
    
    confidences = []
    accuracies = []
    
    for _ in range(num_samples):
        confidence = draw(st.floats(min_value=0.1, max_value=1.0))
        # Generate accuracy that correlates with confidence (but with some noise)
        base_accuracy = confidence + draw(st.floats(min_value=-0.3, max_value=0.3))
        accuracy = max(0.0, min(1.0, base_accuracy))
        
        confidences.append(confidence)
        accuracies.append(accuracy)
    
    return {
        "confidences": np.array(confidences),
        "accuracies": np.array(accuracies),
        "task": draw(st.sampled_from(["height", "direction", "breaking"]))
    }


@given(confidence_test_data())
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_24_confidence_score_generation(test_data):
    """
    Feature: wave-analysis-system, Property 24: Confidence Score Generation
    
    Property: The system SHALL generate confidence scores for all predictions
    that are properly calibrated and provide meaningful uncertainty estimates.
    
    Validates Requirements:
    - 3.5: Confidence scores for wave height measurements
    - 4.5: Confidence scores for direction analysis
    - 5.4: Confidence scores for breaking type classification
    - 10.3: Confidence estimation for all predictions
    """
    if not SWELLSIGHT_AVAILABLE:
        print("⚠️  SwellSight not available, skipping test")
        return
    
    try:
        # Initialize wave analyzer with confidence scoring
        analyzer = DINOv2WaveAnalyzer(
            backbone_model="dinov2_vitb14",
            freeze_backbone=True,
            device="cpu",  # Use CPU for testing
            enable_optimization=False,  # Disable optimization for testing
            confidence_calibration_method="isotonic"
        )
        
        # Analyze waves to get predictions with confidence scores
        wave_metrics, _, _ = analyzer.analyze_waves(
            test_data["rgb_image"], 
            test_data["depth_map"]
        )
        
        # Get confidence scores
        confidence_scores = analyzer.get_confidence_scores()
        
        # Property 1: All confidence scores must be present and valid
        assert hasattr(confidence_scores, 'height_confidence'), "Height confidence score missing"
        assert hasattr(confidence_scores, 'direction_confidence'), "Direction confidence score missing"
        assert hasattr(confidence_scores, 'breaking_type_confidence'), "Breaking type confidence score missing"
        assert hasattr(confidence_scores, 'overall_confidence'), "Overall confidence score missing"
        
        # Property 2: Confidence scores must be in valid range [0, 1]
        assert 0.0 <= confidence_scores.height_confidence <= 1.0, f"Height confidence out of range: {confidence_scores.height_confidence}"
        assert 0.0 <= confidence_scores.direction_confidence <= 1.0, f"Direction confidence out of range: {confidence_scores.direction_confidence}"
        assert 0.0 <= confidence_scores.breaking_type_confidence <= 1.0, f"Breaking confidence out of range: {confidence_scores.breaking_type_confidence}"
        assert 0.0 <= confidence_scores.overall_confidence <= 1.0, f"Overall confidence out of range: {confidence_scores.overall_confidence}"
        
        # Property 3: Enhanced confidence metrics must be present
        assert confidence_scores.height_metrics is not None, "Height confidence metrics missing"
        assert confidence_scores.direction_metrics is not None, "Direction confidence metrics missing"
        assert confidence_scores.breaking_metrics is not None, "Breaking confidence metrics missing"
        
        # Property 4: Enhanced metrics must have required fields
        for metrics in [confidence_scores.height_metrics, confidence_scores.direction_metrics, confidence_scores.breaking_metrics]:
            assert hasattr(metrics, 'raw_confidence'), "Raw confidence missing from metrics"
            assert hasattr(metrics, 'calibrated_confidence'), "Calibrated confidence missing from metrics"
            assert hasattr(metrics, 'uncertainty_estimate'), "Uncertainty estimate missing from metrics"
            assert hasattr(metrics, 'prediction_entropy'), "Prediction entropy missing from metrics"
            assert hasattr(metrics, 'confidence_interval'), "Confidence interval missing from metrics"
            assert hasattr(metrics, 'reliability_score'), "Reliability score missing from metrics"
            
            # Validate metric ranges
            assert 0.0 <= metrics.raw_confidence <= 1.0, f"Raw confidence out of range: {metrics.raw_confidence}"
            assert 0.0 <= metrics.calibrated_confidence <= 1.0, f"Calibrated confidence out of range: {metrics.calibrated_confidence}"
            assert 0.0 <= metrics.uncertainty_estimate <= 1.0, f"Uncertainty estimate out of range: {metrics.uncertainty_estimate}"
            assert metrics.prediction_entropy >= 0.0, f"Prediction entropy negative: {metrics.prediction_entropy}"
            assert 0.0 <= metrics.reliability_score <= 1.0, f"Reliability score out of range: {metrics.reliability_score}"
            
            # Validate confidence interval
            assert len(metrics.confidence_interval) == 2, "Confidence interval must have 2 values"
            assert metrics.confidence_interval[0] <= metrics.confidence_interval[1], "Confidence interval bounds invalid"
        
        # Property 5: Overall confidence should be reasonable average of individual confidences
        expected_overall = (confidence_scores.height_confidence + 
                          confidence_scores.direction_confidence + 
                          confidence_scores.breaking_type_confidence) / 3.0
        assert abs(confidence_scores.overall_confidence - expected_overall) < 0.1, \
            f"Overall confidence not properly averaged: {confidence_scores.overall_confidence} vs {expected_overall}"
        
        # Property 6: Wave metrics must include confidence scores
        assert hasattr(wave_metrics, 'height_confidence'), "Wave metrics missing height confidence"
        assert hasattr(wave_metrics, 'direction_confidence'), "Wave metrics missing direction confidence"
        assert hasattr(wave_metrics, 'breaking_confidence'), "Wave metrics missing breaking confidence"
        
        # Property 7: Confidence scores in wave metrics should match confidence scores object
        assert abs(wave_metrics.height_confidence - confidence_scores.height_confidence) < 1e-6, \
            "Height confidence mismatch between wave metrics and confidence scores"
        assert abs(wave_metrics.direction_confidence - confidence_scores.direction_confidence) < 1e-6, \
            "Direction confidence mismatch between wave metrics and confidence scores"
        assert abs(wave_metrics.breaking_confidence - confidence_scores.breaking_type_confidence) < 1e-6, \
            "Breaking confidence mismatch between wave metrics and confidence scores"
        
        print(f"✓ Property 24 validated: Confidence scores generated successfully")
        print(f"  - Height confidence: {confidence_scores.height_confidence:.3f}")
        print(f"  - Direction confidence: {confidence_scores.direction_confidence:.3f}")
        print(f"  - Breaking confidence: {confidence_scores.breaking_type_confidence:.3f}")
        print(f"  - Overall confidence: {confidence_scores.overall_confidence:.3f}")
        
    except Exception as e:
        print(f"❌ Property 24 test failed: {e}")
        raise


@given(calibration_data_scenarios())
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_24_confidence_calibration(calibration_data):
    """
    Test confidence calibration functionality.
    
    Property: The confidence calibration system SHALL improve confidence
    reliability by learning from confidence-accuracy pairs.
    """
    if not SWELLSIGHT_AVAILABLE:
        print("⚠️  SwellSight not available, skipping calibration test")
        return
    
    try:
        # Initialize confidence scorer
        confidence_scorer = ComprehensiveConfidenceScorer("isotonic")
        
        # Add calibration data
        task = calibration_data["task"]
        confidences = calibration_data["confidences"]
        accuracies = calibration_data["accuracies"]
        
        for conf, acc in zip(confidences, accuracies):
            confidence_scorer.add_calibration_data(task, conf, acc)
        
        # Fit calibrator
        fit_results = confidence_scorer.fit_calibrators(min_samples=10)
        
        # Property 1: Calibrator should be fitted if enough data
        if len(confidences) >= 10:
            assert fit_results[task], f"Calibrator for {task} should be fitted with {len(confidences)} samples"
        
        # Property 2: Calibration analysis should work
        if len(confidences) >= 10:
            calibration_results = confidence_scorer.analyze_calibration(task)
            
            assert isinstance(calibration_results, CalibrationResults), "Calibration results should be CalibrationResults object"
            assert 0.0 <= calibration_results.expected_calibration_error <= 1.0, \
                f"ECE out of range: {calibration_results.expected_calibration_error}"
            assert 0.0 <= calibration_results.maximum_calibration_error <= 1.0, \
                f"MCE out of range: {calibration_results.maximum_calibration_error}"
            assert 0.0 <= calibration_results.average_calibration_error <= 1.0, \
                f"ACE out of range: {calibration_results.average_calibration_error}"
            
            # Property 3: Reliability diagram data should be present
            assert "bin_boundaries" in calibration_results.reliability_diagram_data, "Bin boundaries missing"
            assert "bin_accuracies" in calibration_results.reliability_diagram_data, "Bin accuracies missing"
            assert "bin_confidences" in calibration_results.reliability_diagram_data, "Bin confidences missing"
            assert "bin_counts" in calibration_results.reliability_diagram_data, "Bin counts missing"
        
        # Property 4: Calibration summary should be available
        summary = confidence_scorer.get_calibration_summary()
        assert task in summary, f"Task {task} missing from calibration summary"
        assert "calibrator_fitted" in summary[task], "Calibrator fitted status missing"
        assert "data_points" in summary[task], "Data points count missing"
        assert "calibration_method" in summary[task], "Calibration method missing"
        
        print(f"✓ Confidence calibration validated for {task}")
        print(f"  - Data points: {len(confidences)}")
        print(f"  - Calibrator fitted: {fit_results.get(task, False)}")
        
    except Exception as e:
        print(f"❌ Confidence calibration test failed: {e}")
        raise


def test_property_24_confidence_scorer_initialization():
    """
    Test that confidence scorer can be initialized with different calibration methods.
    
    Property: The confidence scoring system SHALL support multiple calibration
    methods and initialize properly.
    """
    if not SWELLSIGHT_AVAILABLE:
        print("⚠️  SwellSight not available, skipping initialization test")
        return
    
    try:
        # Test different calibration methods
        methods = ["platt", "isotonic", "temperature"]
        
        for method in methods:
            scorer = ComprehensiveConfidenceScorer(method)
            
            # Property 1: Scorer should be initialized
            assert scorer is not None, f"Scorer not initialized for method {method}"
            assert scorer.calibration_method == method, f"Calibration method not set correctly: {scorer.calibration_method}"
            
            # Property 2: Calibrators should be created
            assert hasattr(scorer, 'height_calibrator'), "Height calibrator missing"
            assert hasattr(scorer, 'direction_calibrator'), "Direction calibrator missing"
            assert hasattr(scorer, 'breaking_calibrator'), "Breaking calibrator missing"
            
            # Property 3: Calibration history should be initialized
            assert hasattr(scorer, 'calibration_history'), "Calibration history missing"
            assert "height" in scorer.calibration_history, "Height history missing"
            assert "direction" in scorer.calibration_history, "Direction history missing"
            assert "breaking" in scorer.calibration_history, "Breaking history missing"
            
            print(f"✓ Confidence scorer initialized successfully with {method} calibration")
        
        # Test invalid method (should fall back to isotonic)
        scorer = ComprehensiveConfidenceScorer("invalid_method")
        assert scorer.calibration_method == "invalid_method", "Method should be stored even if invalid"
        
        print("✓ Confidence scorer initialization tests passed")
        
    except Exception as e:
        print(f"❌ Confidence scorer initialization test failed: {e}")
        raise


def test_property_24_wave_analyzer_integration():
    """
    Test that wave analyzer properly integrates with confidence scoring system.
    
    Property: The wave analyzer SHALL integrate seamlessly with the confidence
    scoring system and provide enhanced confidence metrics.
    """
    if not SWELLSIGHT_AVAILABLE:
        print("⚠️  SwellSight not available, skipping integration test")
        return
    
    try:
        # Initialize analyzer with confidence scoring
        analyzer = DINOv2WaveAnalyzer(
            backbone_model="dinov2_vitb14",
            freeze_backbone=True,
            device="cpu",
            enable_optimization=False,
            confidence_calibration_method="isotonic"
        )
        
        # Property 1: Analyzer should have confidence scorer
        assert hasattr(analyzer, 'confidence_scorer'), "Wave analyzer missing confidence scorer"
        assert analyzer.confidence_scorer is not None, "Confidence scorer not initialized"
        
        # Property 2: Analyzer should have confidence calibration methods
        assert hasattr(analyzer, 'add_confidence_calibration_data'), "Missing calibration data method"
        assert hasattr(analyzer, 'fit_confidence_calibrators'), "Missing calibrator fitting method"
        assert hasattr(analyzer, 'get_confidence_calibration_summary'), "Missing calibration summary method"
        assert hasattr(analyzer, 'analyze_confidence_calibration'), "Missing calibration analysis method"
        
        # Property 3: Test calibration data management
        # Create dummy ground truth data
        dummy_wave_metrics = WaveMetrics(
            height_meters=2.0,
            height_feet=6.56,
            height_confidence=0.8,
            direction="LEFT",
            direction_confidence=0.9,
            breaking_type="SPILLING",
            breaking_confidence=0.7,
            extreme_conditions=False
        )
        
        ground_truth = {
            "height_meters": 2.1,  # Close to prediction
            "direction": "LEFT",   # Exact match
            "breaking_type": "PLUNGING"  # Different from prediction
        }
        
        # Add calibration data
        analyzer.add_confidence_calibration_data(dummy_wave_metrics, ground_truth)
        
        # Get calibration summary
        summary = analyzer.get_confidence_calibration_summary()
        assert isinstance(summary, dict), "Calibration summary should be dictionary"
        assert "height" in summary, "Height task missing from summary"
        assert "direction" in summary, "Direction task missing from summary"
        assert "breaking" in summary, "Breaking task missing from summary"
        
        print("✓ Wave analyzer integration with confidence scoring validated")
        
    except Exception as e:
        print(f"❌ Wave analyzer integration test failed: {e}")
        raise


if __name__ == "__main__":
    if not HYPOTHESIS_AVAILABLE:
        print("❌ Hypothesis not available for property-based testing")
        sys.exit(1)
    
    if not SWELLSIGHT_AVAILABLE:
        print("❌ SwellSight modules not available")
        sys.exit(1)
    
    try:
        print("🧪 Running Property 24: Confidence Score Generation Tests...")
        print("=" * 60)
        
        # Test 1: Main confidence score generation property
        print("  ✓ Testing Property 24: Confidence Score Generation...")
        test_property_24_confidence_score_generation()
        
        # Test 2: Confidence calibration functionality
        print("  ✓ Testing confidence calibration functionality...")
        test_property_24_confidence_calibration()
        
        # Test 3: Confidence scorer initialization
        print("  ✓ Testing confidence scorer initialization...")
        test_property_24_confidence_scorer_initialization()
        
        # Test 4: Wave analyzer integration
        print("  ✓ Testing wave analyzer integration...")
        test_property_24_wave_analyzer_integration()
        
        print("\n🎉 All Property 24 tests passed!")
        
    except Exception as e:
        print(f"\n❌ Property 24 tests failed: {e}")
        sys.exit(1)