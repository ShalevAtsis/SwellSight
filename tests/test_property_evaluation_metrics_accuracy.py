"""
Property-based tests for evaluation metrics accuracy.

Tests that evaluation metrics are calculated correctly and consistently
across different input scenarios.
"""

import pytest
import numpy as np
import torch
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

from src.swellsight.evaluation.metrics import WaveAnalysisMetrics, HeightAccuracyMetrics, ClassificationMetrics


class TestEvaluationMetricsAccuracy:
    """Property tests for evaluation metrics accuracy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_calculator = WaveAnalysisMetrics()
    
    @given(
        predictions=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        targets=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_height_metrics_properties(self, predictions, targets):
        """Test properties of height metrics calculation."""
        # Ensure arrays have same length
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        assume(len(predictions) >= 10)  # Need minimum samples
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_height_metrics(predictions, targets)
        
        # Property 1: MAE should be non-negative
        assert metrics.mae >= 0, f"MAE should be non-negative, got {metrics.mae}"
        
        # Property 2: RMSE should be non-negative and >= MAE
        assert metrics.rmse >= 0, f"RMSE should be non-negative, got {metrics.rmse}"
        assert metrics.rmse >= metrics.mae, f"RMSE ({metrics.rmse}) should be >= MAE ({metrics.mae})"
        
        # Property 3: Accuracy percentages should be between 0 and 100
        assert 0 <= metrics.accuracy_within_02m <= 100, f"Accuracy within 0.2m should be 0-100%, got {metrics.accuracy_within_02m}"
        assert 0 <= metrics.accuracy_within_05m <= 100, f"Accuracy within 0.5m should be 0-100%, got {metrics.accuracy_within_05m}"
        
        # Property 4: Accuracy within 0.5m should be >= accuracy within 0.2m
        assert metrics.accuracy_within_05m >= metrics.accuracy_within_02m, \
            f"Accuracy within 0.5m ({metrics.accuracy_within_05m}) should be >= accuracy within 0.2m ({metrics.accuracy_within_02m})"
        
        # Property 5: Perfect predictions should give perfect metrics
        if np.allclose(predictions, targets, atol=1e-6):
            assert metrics.mae < 1e-5, f"Perfect predictions should have MAE ≈ 0, got {metrics.mae}"
            assert metrics.rmse < 1e-5, f"Perfect predictions should have RMSE ≈ 0, got {metrics.rmse}"
    
    @given(
        predictions=arrays(
            dtype=np.int32,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.integers(min_value=0, max_value=2)
        ),
        targets=arrays(
            dtype=np.int32,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.integers(min_value=0, max_value=2)
        ),
        task_type=st.sampled_from(["direction", "breaking"])
    )
    @settings(max_examples=30, deadline=5000)
    def test_classification_metrics_properties(self, predictions, targets, task_type):
        """Test properties of classification metrics calculation."""
        # Ensure arrays have same length
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        assume(len(predictions) >= 20)  # Need minimum samples
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_classification_metrics(
            predictions, targets, task_type
        )
        
        # Property 1: Accuracy should be between 0 and 1
        assert 0 <= metrics.accuracy <= 1, f"Accuracy should be 0-1, got {metrics.accuracy}"
        
        # Property 2: All per-class metrics should be between 0 and 1
        for class_name, precision in metrics.precision_per_class.items():
            assert 0 <= precision <= 1, f"Precision for {class_name} should be 0-1, got {precision}"
        
        for class_name, recall in metrics.recall_per_class.items():
            assert 0 <= recall <= 1, f"Recall for {class_name} should be 0-1, got {recall}"
        
        for class_name, f1 in metrics.f1_score_per_class.items():
            assert 0 <= f1 <= 1, f"F1 for {class_name} should be 0-1, got {f1}"
        
        # Property 3: Macro F1 should be between 0 and 1
        assert 0 <= metrics.macro_avg_f1 <= 1, f"Macro F1 should be 0-1, got {metrics.macro_avg_f1}"
        
        # Property 4: Confusion matrix should have correct shape and non-negative values
        expected_classes = 3  # All tasks have 3 classes
        assert metrics.confusion_matrix.shape == (expected_classes, expected_classes), \
            f"Confusion matrix should be {expected_classes}x{expected_classes}, got {metrics.confusion_matrix.shape}"
        
        assert np.all(metrics.confusion_matrix >= 0), "Confusion matrix should have non-negative values"
        
        # Property 5: Confusion matrix diagonal sum should equal number of correct predictions
        correct_predictions = np.sum(predictions == targets)
        diagonal_sum = np.trace(metrics.confusion_matrix)
        assert diagonal_sum == correct_predictions, \
            f"Confusion matrix diagonal sum ({diagonal_sum}) should equal correct predictions ({correct_predictions})"
        
        # Property 6: Perfect predictions should give perfect metrics
        if np.array_equal(predictions, targets):
            assert metrics.accuracy == 1.0, f"Perfect predictions should have accuracy = 1.0, got {metrics.accuracy}"
            assert metrics.macro_avg_f1 == 1.0, f"Perfect predictions should have macro F1 = 1.0, got {metrics.macro_avg_f1}"
    
    @given(
        confidences=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        accuracies=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.sampled_from([0.0, 1.0])
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_calibration_metrics_properties(self, confidences, accuracies):
        """Test properties of confidence calibration metrics."""
        # Ensure arrays have same length
        min_len = min(len(confidences), len(accuracies))
        confidences = confidences[:min_len]
        accuracies = accuracies[:min_len]
        
        assume(len(confidences) >= 20)  # Need minimum samples
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_calibration_metrics(confidences, accuracies)
        
        # Property 1: ECE should be between 0 and 1
        assert 0 <= metrics.expected_calibration_error <= 1, \
            f"ECE should be 0-1, got {metrics.expected_calibration_error}"
        
        # Property 2: MCE should be between 0 and 1
        assert 0 <= metrics.maximum_calibration_error <= 1, \
            f"MCE should be 0-1, got {metrics.maximum_calibration_error}"
        
        # Property 3: MCE should be >= ECE
        assert metrics.maximum_calibration_error >= metrics.expected_calibration_error, \
            f"MCE ({metrics.maximum_calibration_error}) should be >= ECE ({metrics.expected_calibration_error})"
        
        # Property 4: Reliability diagram data should have consistent structure
        rel_data = metrics.reliability_diagram_data
        if rel_data.get("bin_centers"):
            assert len(rel_data["bin_centers"]) == len(rel_data["bin_accuracies"]), \
                "Reliability diagram data should have consistent lengths"
            assert len(rel_data["bin_centers"]) == len(rel_data["bin_confidences"]), \
                "Reliability diagram data should have consistent lengths"
            assert len(rel_data["bin_centers"]) == len(rel_data["bin_counts"]), \
                "Reliability diagram data should have consistent lengths"
        
        # Property 5: Perfect calibration should give ECE ≈ 0
        if np.allclose(confidences, accuracies, atol=0.1):
            assert metrics.expected_calibration_error < 0.2, \
                f"Well-calibrated predictions should have low ECE, got {metrics.expected_calibration_error}"
    
    @given(
        height_preds=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=15, max_value=50),
            elements=st.floats(min_value=0.1, max_value=8.0, allow_nan=False, allow_infinity=False)
        ),
        direction_preds=arrays(
            dtype=np.int32,
            shape=st.integers(min_value=15, max_value=50),
            elements=st.integers(min_value=0, max_value=2)
        ),
        breaking_preds=arrays(
            dtype=np.int32,
            shape=st.integers(min_value=15, max_value=50),
            elements=st.integers(min_value=0, max_value=2)
        )
    )
    @settings(max_examples=20, deadline=10000)
    def test_complete_metrics_properties(self, height_preds, direction_preds, breaking_preds):
        """Test properties of complete metrics calculation."""
        # Ensure all arrays have same length
        min_len = min(len(height_preds), len(direction_preds), len(breaking_preds))
        height_preds = height_preds[:min_len]
        height_targets = height_preds + np.random.normal(0, 0.1, min_len)  # Add small noise
        direction_preds = direction_preds[:min_len]
        direction_targets = direction_preds.copy()  # Perfect direction predictions
        breaking_preds = breaking_preds[:min_len]
        breaking_targets = breaking_preds.copy()  # Perfect breaking predictions
        
        assume(min_len >= 15)  # Need minimum samples
        
        # Create confidence scores
        confidences = {
            "height": np.random.uniform(0.5, 1.0, min_len),
            "direction": np.random.uniform(0.7, 1.0, min_len),
            "breaking": np.random.uniform(0.6, 1.0, min_len)
        }
        
        # Calculate complete metrics
        metrics = self.metrics_calculator.calculate_complete_metrics(
            height_preds=height_preds,
            height_targets=height_targets,
            direction_preds=direction_preds,
            direction_targets=direction_targets,
            breaking_preds=breaking_preds,
            breaking_targets=breaking_targets,
            confidences=confidences
        )
        
        # Property 1: Overall score should be between 0 and 100
        assert 0 <= metrics.overall_score <= 100, \
            f"Overall score should be 0-100, got {metrics.overall_score}"
        
        # Property 2: All component metrics should be valid
        assert isinstance(metrics.height_metrics, HeightAccuracyMetrics), \
            "Height metrics should be HeightAccuracyMetrics instance"
        assert isinstance(metrics.direction_metrics, ClassificationMetrics), \
            "Direction metrics should be ClassificationMetrics instance"
        assert isinstance(metrics.breaking_type_metrics, ClassificationMetrics), \
            "Breaking type metrics should be ClassificationMetrics instance"
        
        # Property 3: With perfect classification predictions, classification accuracy should be 1.0
        assert metrics.direction_metrics.accuracy == 1.0, \
            f"Perfect direction predictions should have accuracy = 1.0, got {metrics.direction_metrics.accuracy}"
        assert metrics.breaking_type_metrics.accuracy == 1.0, \
            f"Perfect breaking predictions should have accuracy = 1.0, got {metrics.breaking_type_metrics.accuracy}"
        
        # Property 4: Overall score should be reasonable given component scores
        expected_min_score = metrics.direction_metrics.accuracy * 30 + metrics.breaking_type_metrics.accuracy * 30  # 60% from perfect classification
        assert metrics.overall_score >= expected_min_score, \
            f"Overall score ({metrics.overall_score}) should be at least {expected_min_score} with perfect classification"


if __name__ == "__main__":
    pytest.main([__file__])