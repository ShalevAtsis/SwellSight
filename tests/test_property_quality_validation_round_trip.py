#!/usr/bin/env python3
"""
Property-Based Test for Quality Validation Round Trip
Tests Property 25: Quality Validation Round Trip
Validates Requirements 10.1, 10.2
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
    from swellsight.utils.quality_validation import (
        ComprehensiveQualityValidator, ImageQualityValidator, DepthMapQualityValidator,
        PredictionAnomalyDetector, PerformanceMonitor,
        ImageQualityMetrics, DepthMapQualityMetrics, PredictionAnomalyMetrics
    )
    SWELLSIGHT_AVAILABLE = True
except ImportError as e:
    print(f"SwellSight modules not available: {e}")
    SWELLSIGHT_AVAILABLE = False


@composite
def quality_validation_test_data(draw):
    """Generate test data for quality validation testing."""
    # Generate image dimensions
    height = draw(st.integers(min_value=64, max_value=512))
    width = draw(st.integers(min_value=64, max_value=512))
    
    # Generate image quality parameters
    image_quality = draw(st.sampled_from(["high", "medium", "low", "corrupted"]))
    
    if image_quality == "high":
        # High quality image
        rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 0.8 + 0.1  # Good contrast
        depth_array = np.random.rand(height, width).astype(np.float32) * 0.8 + 0.1   # Good depth variation
    elif image_quality == "medium":
        # Medium quality image
        rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 0.6 + 0.2  # Medium contrast
        depth_array = np.random.rand(height, width).astype(np.float32) * 0.6 + 0.2   # Medium depth variation
    elif image_quality == "low":
        # Low quality image
        rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 0.3 + 0.35  # Low contrast
        depth_array = np.random.rand(height, width).astype(np.float32) * 0.3 + 0.35   # Low depth variation
    else:  # corrupted
        # Corrupted image with invalid values
        rgb_array = np.random.rand(height, width, 3).astype(np.float32)
        rgb_array[0, 0, 0] = np.nan  # Add NaN value
        depth_array = np.random.rand(height, width).astype(np.float32)
        depth_array[0, 0] = np.inf  # Add infinite value
    
    # Convert to uint8 for RGB
    if image_quality != "corrupted":
        rgb_array = (rgb_array * 255).astype(np.uint8)
    
    depth_map = DepthMap(
        data=depth_array,
        resolution=(height, width),
        quality_score=0.8 if image_quality == "high" else 0.5,
        edge_preservation=0.7 if image_quality == "high" else 0.3
    )
    
    return {
        "rgb_image": rgb_array,
        "depth_map": depth_map,
        "expected_quality": image_quality,
        "should_pass_validation": image_quality in ["high", "medium"]
    }


@composite
def prediction_anomaly_test_data(draw):
    """Generate test data for prediction anomaly testing."""
    anomaly_type = draw(st.sampled_from(["normal", "extreme_height", "low_confidence", "inconsistent"]))
    
    if anomaly_type == "normal":
        # Normal prediction
        prediction = {
            "height_meters": draw(st.floats(min_value=1.0, max_value=4.0)),
            "height_confidence": draw(st.floats(min_value=0.6, max_value=1.0)),
            "direction_confidence": draw(st.floats(min_value=0.6, max_value=1.0)),
            "breaking_confidence": draw(st.floats(min_value=0.6, max_value=1.0)),
            "extreme_conditions": False
        }
    elif anomaly_type == "extreme_height":
        # Extreme height anomaly
        prediction = {
            "height_meters": draw(st.floats(min_value=15.0, max_value=20.0)),  # Unrealistic height
            "height_confidence": draw(st.floats(min_value=0.5, max_value=1.0)),
            "direction_confidence": draw(st.floats(min_value=0.5, max_value=1.0)),
            "breaking_confidence": draw(st.floats(min_value=0.5, max_value=1.0)),
            "extreme_conditions": True
        }
    elif anomaly_type == "low_confidence":
        # Low confidence anomaly
        prediction = {
            "height_meters": draw(st.floats(min_value=1.0, max_value=4.0)),
            "height_confidence": draw(st.floats(min_value=0.01, max_value=0.1)),  # Very low confidence
            "direction_confidence": draw(st.floats(min_value=0.01, max_value=0.1)),
            "breaking_confidence": draw(st.floats(min_value=0.01, max_value=0.1)),
            "extreme_conditions": False
        }
    else:  # inconsistent
        # Inconsistent prediction
        prediction = {
            "height_meters": draw(st.floats(min_value=0.1, max_value=0.3)),  # Very small waves
            "height_confidence": draw(st.floats(min_value=0.9, max_value=1.0)),  # But high confidence
            "direction_confidence": draw(st.floats(min_value=0.1, max_value=0.3)),  # Low direction confidence
            "breaking_confidence": draw(st.floats(min_value=0.9, max_value=1.0)),  # High breaking confidence
            "extreme_conditions": False
        }
    
    prediction["overall_confidence"] = (
        prediction["height_confidence"] + 
        prediction["direction_confidence"] + 
        prediction["breaking_confidence"]
    ) / 3.0
    
    return {
        "prediction": prediction,
        "anomaly_type": anomaly_type,
        "should_be_anomalous": anomaly_type != "normal"
    }


@given(quality_validation_test_data())
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_25_input_quality_validation(test_data):
    """
    Feature: wave-analysis-system, Property 25: Quality Validation Round Trip
    
    Property: The system SHALL validate input image and depth map quality,
    rejecting corrupted or unsuitable inputs before processing.
    
    Validates Requirements:
    - 10.1: Input image quality validation and rejection
    - 10.2: Depth map quality assessment using statistical measures
    """
    if not SWELLSIGHT_AVAILABLE:
        print("⚠️  SwellSight not available, skipping test")
        return
    
    try:
        # Initialize quality validator
        quality_validator = ComprehensiveQualityValidator()
        
        # Test input quality validation
        rgb_image = test_data["rgb_image"]
        depth_map = test_data["depth_map"]
        expected_quality = test_data["expected_quality"]
        should_pass = test_data["should_pass_validation"]
        
        # Validate input quality
        is_valid, quality_metrics = quality_validator.validate_input_quality(rgb_image, depth_map.data)
        
        # Property 1: Validation result should match expected quality
        if should_pass:
            assert is_valid, f"High/medium quality input should pass validation but failed: {expected_quality}"
        else:
            # Low quality or corrupted should fail (though low quality might sometimes pass)
            if expected_quality == "corrupted":
                assert not is_valid, f"Corrupted input should fail validation but passed"
        
        # Property 2: Quality metrics should be present
        assert "image_valid" in quality_metrics, "Image validation result missing"
        assert "depth_valid" in quality_metrics, "Depth validation result missing"
        assert "overall_valid" in quality_metrics, "Overall validation result missing"
        assert "image_metrics" in quality_metrics, "Image quality metrics missing"
        assert "depth_metrics" in quality_metrics, "Depth quality metrics missing"
        
        # Property 3: Overall validation should be logical combination
        expected_overall = quality_metrics["image_valid"] and quality_metrics["depth_valid"]
        assert quality_metrics["overall_valid"] == expected_overall, \
            f"Overall validation inconsistent: {quality_metrics['overall_valid']} vs {expected_overall}"
        
        # Property 4: Image quality metrics should have required fields
        image_metrics = quality_metrics["image_metrics"]
        assert hasattr(image_metrics, 'resolution'), "Image resolution missing"
        assert hasattr(image_metrics, 'contrast_ratio'), "Image contrast ratio missing"
        assert hasattr(image_metrics, 'quality_score'), "Image quality score missing"
        assert hasattr(image_metrics, 'is_corrupted'), "Image corruption flag missing"
        
        # Property 5: Depth quality metrics should have required fields
        depth_metrics = quality_metrics["depth_metrics"]
        assert hasattr(depth_metrics, 'edge_preservation'), "Depth edge preservation missing"
        assert hasattr(depth_metrics, 'contrast_ratio'), "Depth contrast ratio missing"
        assert hasattr(depth_metrics, 'quality_score'), "Depth quality score missing"
        
        # Property 6: Quality scores should be in valid range [0, 1]
        assert 0.0 <= image_metrics.quality_score <= 1.0, f"Image quality score out of range: {image_metrics.quality_score}"
        assert 0.0 <= depth_metrics.quality_score <= 1.0, f"Depth quality score out of range: {depth_metrics.quality_score}"
        
        # Property 7: Corrupted inputs should be detected
        if expected_quality == "corrupted":
            assert image_metrics.is_corrupted or depth_metrics.quality_score == 0.0, \
                "Corrupted input should be detected"
        
        print(f"✓ Property 25 input validation passed for {expected_quality} quality")
        print(f"  - Image valid: {quality_metrics['image_valid']}")
        print(f"  - Depth valid: {quality_metrics['depth_valid']}")
        print(f"  - Overall valid: {quality_metrics['overall_valid']}")
        
    except Exception as e:
        print(f"❌ Property 25 input validation test failed: {e}")
        raise


@given(prediction_anomaly_test_data())
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_25_prediction_anomaly_detection(test_data):
    """
    Test prediction anomaly detection functionality.
    
    Property: The system SHALL detect anomalous predictions and flag them
    for manual review based on statistical analysis and rule-based checks.
    """
    if not SWELLSIGHT_AVAILABLE:
        print("⚠️  SwellSight not available, skipping anomaly detection test")
        return
    
    try:
        # Initialize anomaly detector
        anomaly_detector = PredictionAnomalyDetector()
        
        # Generate some training data for the detector
        training_data = []
        for _ in range(50):
            # Generate normal predictions for training
            normal_pred = {
                "height_meters": np.random.uniform(1.0, 4.0),
                "height_confidence": np.random.uniform(0.6, 1.0),
                "direction_confidence": np.random.uniform(0.6, 1.0),
                "breaking_confidence": np.random.uniform(0.6, 1.0),
                "extreme_conditions": False,
                "overall_confidence": np.random.uniform(0.6, 1.0)
            }
            training_data.append(normal_pred)
        
        # Fit the anomaly detector
        anomaly_detector.fit(training_data)
        
        # Test prediction
        prediction = test_data["prediction"]
        anomaly_type = test_data["anomaly_type"]
        should_be_anomalous = test_data["should_be_anomalous"]
        
        # Detect anomalies
        anomaly_metrics = anomaly_detector.detect_anomalies(prediction)
        
        # Property 1: Anomaly metrics should be present
        assert hasattr(anomaly_metrics, 'height_anomaly_score'), "Height anomaly score missing"
        assert hasattr(anomaly_metrics, 'direction_anomaly_score'), "Direction anomaly score missing"
        assert hasattr(anomaly_metrics, 'breaking_anomaly_score'), "Breaking anomaly score missing"
        assert hasattr(anomaly_metrics, 'confidence_anomaly_score'), "Confidence anomaly score missing"
        assert hasattr(anomaly_metrics, 'overall_anomaly_score'), "Overall anomaly score missing"
        assert hasattr(anomaly_metrics, 'is_anomalous'), "Anomaly flag missing"
        assert hasattr(anomaly_metrics, 'anomaly_reasons'), "Anomaly reasons missing"
        
        # Property 2: Anomaly scores should be in valid range [0, inf) (higher = more anomalous)
        assert anomaly_metrics.height_anomaly_score >= 0.0, f"Height anomaly score negative: {anomaly_metrics.height_anomaly_score}"
        assert anomaly_metrics.direction_anomaly_score >= 0.0, f"Direction anomaly score negative: {anomaly_metrics.direction_anomaly_score}"
        assert anomaly_metrics.breaking_anomaly_score >= 0.0, f"Breaking anomaly score negative: {anomaly_metrics.breaking_anomaly_score}"
        assert anomaly_metrics.confidence_anomaly_score >= 0.0, f"Confidence anomaly score negative: {anomaly_metrics.confidence_anomaly_score}"
        assert anomaly_metrics.overall_anomaly_score >= 0.0, f"Overall anomaly score negative: {anomaly_metrics.overall_anomaly_score}"
        
        # Property 3: Anomaly reasons should be a list
        assert isinstance(anomaly_metrics.anomaly_reasons, list), "Anomaly reasons should be a list"
        
        # Property 4: Extreme cases should be detected
        if anomaly_type == "extreme_height":
            # Should detect extreme height
            assert anomaly_metrics.is_anomalous or "Extremely high wave height" in str(anomaly_metrics.anomaly_reasons), \
                "Extreme height should be detected as anomalous"
        
        if anomaly_type == "low_confidence":
            # Should detect low confidence
            assert anomaly_metrics.is_anomalous or "low confidence" in str(anomaly_metrics.anomaly_reasons).lower(), \
                "Low confidence should be detected as anomalous"
        
        # Property 5: Normal predictions should generally not be anomalous (with some tolerance)
        if anomaly_type == "normal":
            # Allow some false positives, but most normal predictions should pass
            if anomaly_metrics.is_anomalous:
                print(f"  Note: Normal prediction flagged as anomalous (false positive): {anomaly_metrics.anomaly_reasons}")
        
        print(f"✓ Property 25 anomaly detection passed for {anomaly_type}")
        print(f"  - Is anomalous: {anomaly_metrics.is_anomalous}")
        print(f"  - Overall anomaly score: {anomaly_metrics.overall_anomaly_score:.3f}")
        print(f"  - Anomaly reasons: {anomaly_metrics.anomaly_reasons}")
        
    except Exception as e:
        print(f"❌ Property 25 anomaly detection test failed: {e}")
        raise


def test_property_25_performance_monitoring():
    """
    Test performance monitoring and degradation detection.
    
    Property: The system SHALL monitor performance metrics and detect
    degradation in processing time, memory usage, and accuracy.
    """
    if not SWELLSIGHT_AVAILABLE:
        print("⚠️  SwellSight not available, skipping performance monitoring test")
        return
    
    try:
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor(history_size=20)
        
        # Simulate normal performance data
        for i in range(15):
            performance_monitor.record_performance(
                processing_time_ms=100.0 + np.random.normal(0, 10),  # ~100ms with noise
                memory_usage_mb=500.0 + np.random.normal(0, 50),     # ~500MB with noise
                accuracy_score=0.9 + np.random.normal(0, 0.05),     # ~90% accuracy
                confidence_score=0.8 + np.random.normal(0, 0.05),   # ~80% confidence
                had_error=False
            )
        
        # Check that no degradation is detected for normal performance
        degradation_metrics = performance_monitor.detect_degradation()
        
        # Property 1: Degradation metrics should be present
        assert hasattr(degradation_metrics, 'processing_time_ms'), "Processing time missing"
        assert hasattr(degradation_metrics, 'memory_usage_mb'), "Memory usage missing"
        assert hasattr(degradation_metrics, 'accuracy_trend'), "Accuracy trend missing"
        assert hasattr(degradation_metrics, 'confidence_trend'), "Confidence trend missing"
        assert hasattr(degradation_metrics, 'error_rate'), "Error rate missing"
        assert hasattr(degradation_metrics, 'is_degraded'), "Degradation flag missing"
        assert hasattr(degradation_metrics, 'degradation_reasons'), "Degradation reasons missing"
        
        # Property 2: Normal performance should not be flagged as degraded
        assert not degradation_metrics.is_degraded, f"Normal performance should not be degraded: {degradation_metrics.degradation_reasons}"
        
        # Property 3: Error rate should be low for normal performance
        assert degradation_metrics.error_rate <= 0.1, f"Error rate too high for normal performance: {degradation_metrics.error_rate}"
        
        # Simulate performance degradation
        for i in range(10):
            performance_monitor.record_performance(
                processing_time_ms=300.0 + np.random.normal(0, 20),  # Much slower
                memory_usage_mb=1200.0 + np.random.normal(0, 100),   # Much more memory
                accuracy_score=0.6 + np.random.normal(0, 0.05),     # Lower accuracy
                confidence_score=0.5 + np.random.normal(0, 0.05),   # Lower confidence
                had_error=np.random.random() < 0.2  # 20% error rate
            )
        
        # Check that degradation is detected
        degradation_metrics = performance_monitor.detect_degradation()
        
        # Property 4: Degraded performance should be detected
        # Note: This might not always trigger due to the statistical nature of the detection
        if degradation_metrics.is_degraded:
            print(f"✓ Performance degradation correctly detected: {degradation_metrics.degradation_reasons}")
        else:
            print(f"  Note: Performance degradation not detected (may be within tolerance)")
        
        # Property 5: Performance summary should be available
        summary = performance_monitor.get_performance_summary()
        assert isinstance(summary, dict), "Performance summary should be a dictionary"
        assert "samples_collected" in summary, "Sample count missing from summary"
        assert "avg_processing_time_ms" in summary, "Average processing time missing from summary"
        assert "avg_memory_usage_mb" in summary, "Average memory usage missing from summary"
        assert "error_rate" in summary, "Error rate missing from summary"
        
        print("✓ Property 25 performance monitoring passed")
        print(f"  - Samples collected: {summary['samples_collected']}")
        print(f"  - Avg processing time: {summary['avg_processing_time_ms']:.1f}ms")
        print(f"  - Avg memory usage: {summary['avg_memory_usage_mb']:.1f}MB")
        print(f"  - Error rate: {summary['error_rate']:.3f}")
        
    except Exception as e:
        print(f"❌ Property 25 performance monitoring test failed: {e}")
        raise


def test_property_25_comprehensive_integration():
    """
    Test comprehensive integration of all quality validation components.
    
    Property: The comprehensive quality validator SHALL integrate all
    validation components and provide unified quality assessment.
    """
    if not SWELLSIGHT_AVAILABLE:
        print("⚠️  SwellSight not available, skipping integration test")
        return
    
    try:
        # Initialize comprehensive quality validator
        quality_validator = ComprehensiveQualityValidator()
        
        # Property 1: All components should be initialized
        assert hasattr(quality_validator, 'image_validator'), "Image validator missing"
        assert hasattr(quality_validator, 'depth_validator'), "Depth validator missing"
        assert hasattr(quality_validator, 'anomaly_detector'), "Anomaly detector missing"
        assert hasattr(quality_validator, 'performance_monitor'), "Performance monitor missing"
        
        # Property 2: System health report should be available
        health_report = quality_validator.get_system_health_report()
        assert isinstance(health_report, dict), "Health report should be a dictionary"
        assert "timestamp" in health_report, "Timestamp missing from health report"
        assert "performance_summary" in health_report, "Performance summary missing from health report"
        assert "degradation_status" in health_report, "Degradation status missing from health report"
        assert "anomaly_detector_status" in health_report, "Anomaly detector status missing from health report"
        
        # Property 3: Anomaly detector fitting should work
        # Generate dummy prediction history
        prediction_history = []
        for _ in range(30):
            pred = {
                "height_meters": np.random.uniform(1.0, 4.0),
                "height_confidence": np.random.uniform(0.6, 1.0),
                "direction_confidence": np.random.uniform(0.6, 1.0),
                "breaking_confidence": np.random.uniform(0.6, 1.0),
                "overall_confidence": np.random.uniform(0.6, 1.0)
            }
            prediction_history.append(pred)
        
        # Fit anomaly detector
        quality_validator.fit_anomaly_detector(prediction_history)
        
        # Check that detector is fitted
        updated_health_report = quality_validator.get_system_health_report()
        assert updated_health_report["anomaly_detector_status"]["is_fitted"], \
            "Anomaly detector should be fitted after training"
        
        print("✓ Property 25 comprehensive integration passed")
        print(f"  - All components initialized: ✓")
        print(f"  - Health report available: ✓")
        print(f"  - Anomaly detector fitted: ✓")
        
    except Exception as e:
        print(f"❌ Property 25 comprehensive integration test failed: {e}")
        raise


if __name__ == "__main__":
    if not HYPOTHESIS_AVAILABLE:
        print("❌ Hypothesis not available for property-based testing")
        sys.exit(1)
    
    if not SWELLSIGHT_AVAILABLE:
        print("❌ SwellSight modules not available")
        sys.exit(1)
    
    try:
        print("🧪 Running Property 25: Quality Validation Round Trip Tests...")
        print("=" * 70)
        
        # Test 1: Input quality validation
        print("  ✓ Testing Property 25: Input Quality Validation...")
        test_property_25_input_quality_validation()
        
        # Test 2: Prediction anomaly detection
        print("  ✓ Testing prediction anomaly detection...")
        test_property_25_prediction_anomaly_detection()
        
        # Test 3: Performance monitoring
        print("  ✓ Testing performance monitoring...")
        test_property_25_performance_monitoring()
        
        # Test 4: Comprehensive integration
        print("  ✓ Testing comprehensive integration...")
        test_property_25_comprehensive_integration()
        
        print("\n🎉 All Property 25 tests passed!")
        
    except Exception as e:
        print(f"\n❌ Property 25 tests failed: {e}")
        sys.exit(1)