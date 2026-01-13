"""
Wave analysis metrics for model evaluation.

Implements accuracy metrics for wave height, direction classification,
and breaking type classification with confidence calibration.
"""

from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

@dataclass
class HeightAccuracyMetrics:
    """Accuracy metrics for wave height prediction."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    accuracy_within_02m: float  # Percentage within ±0.2m
    accuracy_within_05m: float  # Percentage within ±0.5m
    extreme_condition_detection: float

@dataclass
class ClassificationMetrics:
    """Metrics for classification tasks (direction, breaking type)."""
    accuracy: float
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_score_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    macro_avg_f1: float

@dataclass
class CalibrationMetrics:
    """Confidence calibration metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    reliability_diagram_data: Dict[str, List[float]]

@dataclass
class AccuracyMetrics:
    """Complete accuracy metrics for all wave analysis tasks."""
    height_metrics: HeightAccuracyMetrics
    direction_metrics: ClassificationMetrics
    breaking_type_metrics: ClassificationMetrics
    confidence_calibration: CalibrationMetrics
    overall_score: float

class WaveAnalysisMetrics:
    """Comprehensive metrics calculator for wave analysis evaluation."""
    
    def __init__(self):
        """Initialize wave analysis metrics calculator."""
        self.class_names = {
            "direction": ["LEFT", "RIGHT", "STRAIGHT"],
            "breaking": ["SPILLING", "PLUNGING", "SURGING"]
        }
    
    def calculate_height_metrics(self, 
                                predictions: np.ndarray,
                                targets: np.ndarray) -> HeightAccuracyMetrics:
        """Calculate wave height accuracy metrics.
        
        Args:
            predictions: Predicted wave heights in meters
            targets: Ground truth wave heights in meters
            
        Returns:
            HeightAccuracyMetrics with detailed accuracy assessment
        """
        # Calculate absolute errors
        absolute_errors = np.abs(predictions - targets)
        
        # Mean Absolute Error
        mae = np.mean(absolute_errors)
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # Accuracy within ±0.2m
        accuracy_02m = np.mean(absolute_errors <= 0.2) * 100
        
        # Accuracy within ±0.5m
        accuracy_05m = np.mean(absolute_errors <= 0.5) * 100
        
        # Extreme condition detection (waves < 0.5m or > 8.0m)
        extreme_mask = (targets < 0.5) | (targets > 8.0)
        if np.any(extreme_mask):
            extreme_predictions = predictions[extreme_mask]
            extreme_targets = targets[extreme_mask]
            extreme_detection = np.mean(
                ((extreme_predictions < 0.5) & (extreme_targets < 0.5)) |
                ((extreme_predictions > 8.0) & (extreme_targets > 8.0))
            ) * 100
        else:
            extreme_detection = 100.0  # No extreme conditions to detect
        
        return HeightAccuracyMetrics(
            mae=mae,
            rmse=rmse,
            accuracy_within_02m=accuracy_02m,
            accuracy_within_05m=accuracy_05m,
            extreme_condition_detection=extreme_detection
        )
    
    def calculate_classification_metrics(self,
                                       predictions: np.ndarray,
                                       targets: np.ndarray,
                                       task_type: str) -> ClassificationMetrics:
        """Calculate classification metrics for direction or breaking type.
        
        Args:
            predictions: Predicted class indices
            targets: Ground truth class indices
            task_type: "direction" or "breaking"
            
        Returns:
            ClassificationMetrics with detailed classification assessment
        """
        # Calculate basic accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Get class names for this task
        class_names = self.class_names[task_type]
        
        # Calculate precision, recall, F1-score per class
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0, labels=list(range(len(class_names)))
        )
        
        # Create per-class dictionaries (handle missing classes)
        precision_per_class = {}
        recall_per_class = {}
        f1_per_class = {}
        
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                precision_per_class[class_name] = precision[i]
                recall_per_class[class_name] = recall[i]
                f1_per_class[class_name] = f1[i]
            else:
                # Class not present in data
                precision_per_class[class_name] = 0.0
                recall_per_class[class_name] = 0.0
                f1_per_class[class_name] = 0.0
        
        # Calculate macro-averaged F1
        macro_avg_f1 = np.mean(f1)
        
        # Calculate confusion matrix with all expected classes
        conf_matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision_per_class=precision_per_class,
            recall_per_class=recall_per_class,
            f1_score_per_class=f1_per_class,
            confusion_matrix=conf_matrix,
            macro_avg_f1=macro_avg_f1
        )
    
    def calculate_calibration_metrics(self,
                                    confidences: np.ndarray,
                                    accuracies: np.ndarray) -> CalibrationMetrics:
        """Calculate confidence calibration metrics.
        
        Args:
            confidences: Predicted confidence scores
            accuracies: Binary accuracy indicators
            
        Returns:
            CalibrationMetrics with calibration assessment
        """
        # Sort by confidence
        sorted_indices = np.argsort(confidences)
        sorted_confidences = confidences[sorted_indices]
        sorted_accuracies = accuracies[sorted_indices]
        
        # Calculate Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        reliability_data = {"bin_centers": [], "bin_accuracies": [], "bin_confidences": [], "bin_counts": []}
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (sorted_confidences > bin_lower) & (sorted_confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                accuracy_in_bin = sorted_accuracies[in_bin].mean()
                avg_confidence_in_bin = sorted_confidences[in_bin].mean()
                
                # Update ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                # Update MCE
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
                # Store reliability diagram data
                reliability_data["bin_centers"].append((bin_lower + bin_upper) / 2)
                reliability_data["bin_accuracies"].append(accuracy_in_bin)
                reliability_data["bin_confidences"].append(avg_confidence_in_bin)
                reliability_data["bin_counts"].append(in_bin.sum())
        
        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            reliability_diagram_data=reliability_data
        )
    
    def calculate_complete_metrics(self,
                                 height_preds: np.ndarray,
                                 height_targets: np.ndarray,
                                 direction_preds: np.ndarray,
                                 direction_targets: np.ndarray,
                                 breaking_preds: np.ndarray,
                                 breaking_targets: np.ndarray,
                                 confidences: Dict[str, np.ndarray]) -> AccuracyMetrics:
        """Calculate complete metrics for all wave analysis tasks.
        
        Args:
            height_preds: Wave height predictions
            height_targets: Wave height ground truth
            direction_preds: Direction predictions
            direction_targets: Direction ground truth
            breaking_preds: Breaking type predictions
            breaking_targets: Breaking type ground truth
            confidences: Confidence scores for all tasks
            
        Returns:
            AccuracyMetrics with comprehensive evaluation
        """
        # Calculate individual task metrics
        height_metrics = self.calculate_height_metrics(height_preds, height_targets)
        direction_metrics = self.calculate_classification_metrics(
            direction_preds, direction_targets, "direction"
        )
        breaking_metrics = self.calculate_classification_metrics(
            breaking_preds, breaking_targets, "breaking"
        )
        
        # Calculate confidence calibration
        # Combine all confidence scores and accuracies for calibration
        all_confidences = []
        all_accuracies = []
        
        # Height task calibration (use accuracy within 0.2m as binary accuracy)
        if "height" in confidences:
            height_accuracies = (np.abs(height_preds - height_targets) <= 0.2).astype(float)
            all_confidences.extend(confidences["height"])
            all_accuracies.extend(height_accuracies)
        
        # Direction task calibration
        if "direction" in confidences:
            direction_accuracies = (direction_preds == direction_targets).astype(float)
            all_confidences.extend(confidences["direction"])
            all_accuracies.extend(direction_accuracies)
        
        # Breaking type task calibration
        if "breaking" in confidences:
            breaking_accuracies = (breaking_preds == breaking_targets).astype(float)
            all_confidences.extend(confidences["breaking"])
            all_accuracies.extend(breaking_accuracies)
        
        if all_confidences:
            calibration_metrics = self.calculate_calibration_metrics(
                np.array(all_confidences), np.array(all_accuracies)
            )
        else:
            calibration_metrics = CalibrationMetrics(
                expected_calibration_error=0.0,
                maximum_calibration_error=0.0,
                reliability_diagram_data={}
            )
        
        # Calculate overall score
        overall_score = (
            height_metrics.accuracy_within_02m * 0.4 +
            direction_metrics.accuracy * 0.3 +
            breaking_metrics.accuracy * 0.3
        )
        
        return AccuracyMetrics(
            height_metrics=height_metrics,
            direction_metrics=direction_metrics,
            breaking_type_metrics=breaking_metrics,
            confidence_calibration=calibration_metrics,
            overall_score=overall_score
        )