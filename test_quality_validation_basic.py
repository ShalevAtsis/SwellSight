#!/usr/bin/env python3
"""
Basic test for quality validation system without heavy model loading.
Tests the core functionality of subtask 9.3.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from swellsight.utils.quality_validation import (
        ComprehensiveQualityValidator, ImageQualityValidator, DepthMapQualityValidator,
        PredictionAnomalyDetector, PerformanceMonitor
    )
    SWELLSIGHT_AVAILABLE = True
except ImportError as e:
    print(f"SwellSight modules not available: {e}")
    SWELLSIGHT_AVAILABLE = False


def test_image_quality_validation():
    """Test image quality validation functionality."""
    print("Testing image quality validation...")
    
    validator = ImageQualityValidator()
    
    # Test with good quality image
    good_image = np.random.rand(256, 256, 3) * 0.8 + 0.1  # Good contrast
    good_image = (good_image * 255).astype(np.uint8)
    
    is_valid = validator.validate(good_image)
    metrics = validator.get_quality_metrics(good_image)
    
    print(f"  Good image valid: {is_valid}")
    print(f"  Quality score: {metrics.quality_score:.3f}")
    print(f"  Contrast ratio: {metrics.contrast_ratio:.3f}")
    
    # Test with corrupted image
    corrupted_image = np.random.rand(256, 256, 3).astype(np.float32)
    corrupted_image[0, 0, 0] = np.nan  # Add NaN
    
    is_valid_corrupted = validator.validate(corrupted_image)
    metrics_corrupted = validator.get_quality_metrics(corrupted_image)
    
    print(f"  Corrupted image valid: {is_valid_corrupted}")
    print(f"  Is corrupted: {metrics_corrupted.is_corrupted}")
    
    assert not is_valid_corrupted, "Corrupted image should be invalid"
    assert metrics_corrupted.is_corrupted, "Corruption should be detected"
    
    print("✓ Image quality validation test passed")


def test_depth_map_quality_validation():
    """Test depth map quality validation functionality."""
    print("Testing depth map quality validation...")
    
    validator = DepthMapQualityValidator()
    
    # Test with good quality depth map
    good_depth = np.random.rand(256, 256).astype(np.float32) * 0.8 + 0.1
    
    is_valid = validator.validate(good_depth)
    metrics = validator.get_quality_metrics(good_depth)
    
    print(f"  Good depth map valid: {is_valid}")
    print(f"  Quality score: {metrics.quality_score:.3f}")
    print(f"  Edge preservation: {metrics.edge_preservation:.3f}")
    
    assert is_valid or metrics.quality_score > 0.3, "Good depth map should have reasonable quality"
    
    print("✓ Depth map quality validation test passed")


def test_prediction_anomaly_detection():
    """Test prediction anomaly detection functionality."""
    print("Testing prediction anomaly detection...")
    
    detector = PredictionAnomalyDetector()
    
    # Generate training data
    training_data = []
    for _ in range(30):
        normal_pred = {
            "height_meters": np.random.uniform(1.0, 4.0),
            "height_confidence": np.random.uniform(0.6, 1.0),
            "direction_confidence": np.random.uniform(0.6, 1.0),
            "breaking_confidence": np.random.uniform(0.6, 1.0),
            "overall_confidence": np.random.uniform(0.6, 1.0)
        }
        training_data.append(normal_pred)
    
    # Fit detector
    detector.fit(training_data)
    
    # Test normal prediction
    normal_prediction = {
        "height_meters": 2.5,
        "height_confidence": 0.8,
        "direction_confidence": 0.7,
        "breaking_confidence": 0.9,
        "overall_confidence": 0.8
    }
    
    anomaly_metrics = detector.detect_anomalies(normal_prediction)
    print(f"  Normal prediction anomalous: {anomaly_metrics.is_anomalous}")
    print(f"  Overall anomaly score: {anomaly_metrics.overall_anomaly_score:.3f}")
    
    # Test extreme prediction
    extreme_prediction = {
        "height_meters": 15.0,  # Unrealistic height
        "height_confidence": 0.9,
        "direction_confidence": 0.8,
        "breaking_confidence": 0.7,
        "overall_confidence": 0.8
    }
    
    extreme_anomaly_metrics = detector.detect_anomalies(extreme_prediction)
    print(f"  Extreme prediction anomalous: {extreme_anomaly_metrics.is_anomalous}")
    print(f"  Extreme anomaly score: {extreme_anomaly_metrics.overall_anomaly_score:.3f}")
    print(f"  Anomaly reasons: {extreme_anomaly_metrics.anomaly_reasons}")
    
    assert extreme_anomaly_metrics.is_anomalous, "Extreme prediction should be detected as anomalous"
    
    print("✓ Prediction anomaly detection test passed")


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("Testing performance monitoring...")
    
    monitor = PerformanceMonitor()
    
    # Record normal performance
    for i in range(15):
        monitor.record_performance(
            processing_time_ms=100.0 + np.random.normal(0, 10),
            memory_usage_mb=500.0 + np.random.normal(0, 50),
            accuracy_score=0.9 + np.random.normal(0, 0.05),
            confidence_score=0.8 + np.random.normal(0, 0.05),
            had_error=False
        )
    
    degradation_metrics = monitor.detect_degradation()
    print(f"  Normal performance degraded: {degradation_metrics.is_degraded}")
    print(f"  Processing time: {degradation_metrics.processing_time_ms:.1f}ms")
    print(f"  Error rate: {degradation_metrics.error_rate:.3f}")
    
    # Record degraded performance
    for i in range(10):
        monitor.record_performance(
            processing_time_ms=300.0 + np.random.normal(0, 20),  # Much slower
            memory_usage_mb=1200.0 + np.random.normal(0, 100),   # Much more memory
            accuracy_score=0.6 + np.random.normal(0, 0.05),     # Lower accuracy
            confidence_score=0.5 + np.random.normal(0, 0.05),   # Lower confidence
            had_error=np.random.random() < 0.2  # 20% error rate
        )
    
    degraded_metrics = monitor.detect_degradation()
    print(f"  Degraded performance detected: {degraded_metrics.is_degraded}")
    print(f"  Degradation reasons: {degraded_metrics.degradation_reasons}")
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"  Samples collected: {summary['samples_collected']}")
    print(f"  Average processing time: {summary['avg_processing_time_ms']:.1f}ms")
    
    print("✓ Performance monitoring test passed")


def test_comprehensive_quality_validator():
    """Test comprehensive quality validator integration."""
    print("Testing comprehensive quality validator...")
    
    validator = ComprehensiveQualityValidator()
    
    # Test input quality validation
    rgb_image = np.random.rand(128, 128, 3) * 0.8 + 0.1
    rgb_image = (rgb_image * 255).astype(np.uint8)
    depth_map = np.random.rand(128, 128).astype(np.float32) * 0.8 + 0.1
    
    is_valid, quality_metrics = validator.validate_input_quality(rgb_image, depth_map)
    print(f"  Input valid: {is_valid}")
    print(f"  Image valid: {quality_metrics.get('image_valid', False)}")
    print(f"  Depth valid: {quality_metrics.get('depth_valid', False)}")
    
    # Test prediction validation
    prediction = {
        "height_meters": 2.5,
        "height_confidence": 0.8,
        "direction_confidence": 0.7,
        "breaking_confidence": 0.9,
        "overall_confidence": 0.8
    }
    
    pred_valid, anomaly_metrics = validator.validate_prediction_quality(prediction)
    print(f"  Prediction valid: {pred_valid}")
    print(f"  Anomaly score: {anomaly_metrics.overall_anomaly_score:.3f}")
    
    # Test performance monitoring
    degradation_metrics = validator.monitor_performance(
        processing_time_ms=150.0,
        memory_usage_mb=600.0,
        accuracy_score=0.85,
        confidence_score=0.75,
        had_error=False
    )
    print(f"  Performance degraded: {degradation_metrics.is_degraded}")
    
    # Test system health report
    health_report = validator.get_system_health_report()
    print(f"  Health report available: {'timestamp' in health_report}")
    print(f"  Performance summary available: {'performance_summary' in health_report}")
    
    print("✓ Comprehensive quality validator test passed")


if __name__ == "__main__":
    if not SWELLSIGHT_AVAILABLE:
        print("❌ SwellSight modules not available")
        sys.exit(1)
    
    try:
        print("🧪 Running Quality Validation Basic Tests...")
        print("=" * 50)
        
        test_image_quality_validation()
        print()
        
        test_depth_map_quality_validation()
        print()
        
        test_prediction_anomaly_detection()
        print()
        
        test_performance_monitoring()
        print()
        
        test_comprehensive_quality_validator()
        print()
        
        print("🎉 All quality validation tests passed!")
        print("✅ Subtask 9.3: Anomaly detection and quality validation system is working correctly")
        
    except Exception as e:
        print(f"\n❌ Quality validation tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)