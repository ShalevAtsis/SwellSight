"""
Comprehensive confidence scoring and calibration system for wave analysis predictions.

This module provides advanced confidence estimation, calibration, and uncertainty
quantification across all wave analysis tasks (height, direction, breaking type).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics for model predictions."""
    raw_confidence: float
    calibrated_confidence: float
    uncertainty_estimate: float
    prediction_entropy: float
    confidence_interval: Tuple[float, float]
    reliability_score: float
    
@dataclass
class CalibrationResults:
    """Results from confidence calibration analysis."""
    expected_calibration_error: float
    maximum_calibration_error: float
    average_calibration_error: float
    reliability_diagram_data: Dict[str, np.ndarray]
    calibration_curve_data: Tuple[np.ndarray, np.ndarray]
    is_well_calibrated: bool
    
@dataclass
class UncertaintyQuantification:
    """Uncertainty quantification results."""
    aleatoric_uncertainty: float  # Data uncertainty
    epistemic_uncertainty: float  # Model uncertainty
    total_uncertainty: float
    confidence_interval_95: Tuple[float, float]
    prediction_variance: float

class ConfidenceCalibrator(ABC):
    """Abstract base class for confidence calibration methods."""
    
    @abstractmethod
    def fit(self, confidences: np.ndarray, accuracies: np.ndarray) -> None:
        """Fit calibration model to confidence-accuracy pairs."""
        pass
    
    @abstractmethod
    def calibrate(self, confidence: float) -> float:
        """Calibrate a single confidence score."""
        pass
    
    @abstractmethod
    def calibrate_batch(self, confidences: np.ndarray) -> np.ndarray:
        """Calibrate a batch of confidence scores."""
        pass

class PlattScalingCalibrator(ConfidenceCalibrator):
    """Platt scaling calibration using logistic regression."""
    
    def __init__(self):
        self.calibrator = LogisticRegression()
        self.is_fitted = False
    
    def fit(self, confidences: np.ndarray, accuracies: np.ndarray) -> None:
        """Fit Platt scaling calibrator."""
        if len(confidences) < 10:
            logger.warning("Insufficient data for Platt scaling calibration")
            return
        
        # Reshape for sklearn
        X = confidences.reshape(-1, 1)
        y = accuracies
        
        try:
            self.calibrator.fit(X, y)
            self.is_fitted = True
            logger.info("Platt scaling calibrator fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit Platt scaling calibrator: {e}")
            self.is_fitted = False
    
    def calibrate(self, confidence: float) -> float:
        """Calibrate single confidence score."""
        if not self.is_fitted:
            return confidence
        
        try:
            calibrated = self.calibrator.predict_proba([[confidence]])[0, 1]
            return float(np.clip(calibrated, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Calibration failed, returning original confidence: {e}")
            return confidence
    
    def calibrate_batch(self, confidences: np.ndarray) -> np.ndarray:
        """Calibrate batch of confidence scores."""
        if not self.is_fitted:
            return confidences
        
        try:
            X = confidences.reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(X)[:, 1]
            return np.clip(calibrated, 0.0, 1.0)
        except Exception as e:
            logger.warning(f"Batch calibration failed, returning original confidences: {e}")
            return confidences

class IsotonicCalibrator(ConfidenceCalibrator):
    """Isotonic regression calibration."""
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
    
    def fit(self, confidences: np.ndarray, accuracies: np.ndarray) -> None:
        """Fit isotonic regression calibrator."""
        if len(confidences) < 10:
            logger.warning("Insufficient data for isotonic calibration")
            return
        
        try:
            self.calibrator.fit(confidences, accuracies)
            self.is_fitted = True
            logger.info("Isotonic calibrator fitted successfully")
        except Exception as e:
            logger.error(f"Failed to fit isotonic calibrator: {e}")
            self.is_fitted = False
    
    def calibrate(self, confidence: float) -> float:
        """Calibrate single confidence score."""
        if not self.is_fitted:
            return confidence
        
        try:
            calibrated = self.calibrator.predict([confidence])[0]
            return float(np.clip(calibrated, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Calibration failed, returning original confidence: {e}")
            return confidence
    
    def calibrate_batch(self, confidences: np.ndarray) -> np.ndarray:
        """Calibrate batch of confidence scores."""
        if not self.is_fitted:
            return confidences
        
        try:
            calibrated = self.calibrator.predict(confidences)
            return np.clip(calibrated, 0.0, 1.0)
        except Exception as e:
            logger.warning(f"Batch calibration failed, returning original confidences: {e}")
            return confidences

class TemperatureScalingCalibrator(ConfidenceCalibrator):
    """Temperature scaling calibration for neural network outputs."""
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_fitted = False
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> None:
        """Fit temperature scaling parameter."""
        if len(logits) < 10:
            logger.warning("Insufficient data for temperature scaling")
            return
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        try:
            optimizer.step(eval_loss)
            self.is_fitted = True
            logger.info(f"Temperature scaling fitted with T={self.temperature.item():.3f}")
        except Exception as e:
            logger.error(f"Failed to fit temperature scaling: {e}")
            self.is_fitted = False
    
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrate logits using temperature scaling."""
        if not self.is_fitted:
            return logits
        
        return logits / self.temperature
    
    def calibrate(self, confidence: float) -> float:
        """Calibrate single confidence score (not directly applicable for temperature scaling)."""
        return confidence
    
    def calibrate_batch(self, confidences: np.ndarray) -> np.ndarray:
        """Calibrate batch of confidence scores (not directly applicable for temperature scaling)."""
        return confidences

class ComprehensiveConfidenceScorer:
    """Comprehensive confidence scoring system with calibration and uncertainty quantification."""
    
    def __init__(self, calibration_method: str = "isotonic"):
        """Initialize confidence scorer.
        
        Args:
            calibration_method: Method for calibration ("platt", "isotonic", "temperature")
        """
        self.calibration_method = calibration_method
        self.height_calibrator = self._create_calibrator()
        self.direction_calibrator = self._create_calibrator()
        self.breaking_calibrator = self._create_calibrator()
        self.temperature_calibrator = TemperatureScalingCalibrator()
        
        # Calibration history for analysis
        self.calibration_history = {
            "height": {"confidences": [], "accuracies": []},
            "direction": {"confidences": [], "accuracies": []},
            "breaking": {"confidences": [], "accuracies": []}
        }
        
        logger.info(f"Initialized ComprehensiveConfidenceScorer with {calibration_method} calibration")
    
    def _create_calibrator(self) -> ConfidenceCalibrator:
        """Create calibrator based on specified method."""
        if self.calibration_method == "platt":
            return PlattScalingCalibrator()
        elif self.calibration_method == "isotonic":
            return IsotonicCalibrator()
        else:
            logger.warning(f"Unknown calibration method {self.calibration_method}, using isotonic")
            return IsotonicCalibrator()
    
    def add_calibration_data(self, task: str, confidence: float, accuracy: float) -> None:
        """Add calibration data point for a specific task.
        
        Args:
            task: Task name ("height", "direction", "breaking")
            confidence: Predicted confidence score
            accuracy: Actual accuracy (1.0 if correct, 0.0 if incorrect)
        """
        if task in self.calibration_history:
            self.calibration_history[task]["confidences"].append(confidence)
            self.calibration_history[task]["accuracies"].append(accuracy)
    
    def fit_calibrators(self, min_samples: int = 50) -> Dict[str, bool]:
        """Fit all calibrators using accumulated data.
        
        Args:
            min_samples: Minimum samples required for calibration
            
        Returns:
            Dictionary indicating which calibrators were successfully fitted
        """
        results = {}
        
        for task in ["height", "direction", "breaking"]:
            confidences = np.array(self.calibration_history[task]["confidences"])
            accuracies = np.array(self.calibration_history[task]["accuracies"])
            
            if len(confidences) >= min_samples:
                calibrator = getattr(self, f"{task}_calibrator")
                calibrator.fit(confidences, accuracies)
                results[task] = calibrator.is_fitted
                logger.info(f"Fitted {task} calibrator with {len(confidences)} samples")
            else:
                results[task] = False
                logger.warning(f"Insufficient data for {task} calibrator: {len(confidences)}/{min_samples}")
        
        return results
    
    def compute_confidence_metrics(self, raw_confidence: float, task: str, 
                                 prediction_probs: Optional[np.ndarray] = None) -> ConfidenceMetrics:
        """Compute comprehensive confidence metrics for a prediction.
        
        Args:
            raw_confidence: Raw confidence from model
            task: Task name ("height", "direction", "breaking")
            prediction_probs: Probability distribution over classes (for classification tasks)
            
        Returns:
            ConfidenceMetrics with comprehensive confidence analysis
        """
        # Get appropriate calibrator
        calibrator = getattr(self, f"{task}_calibrator", None)
        
        # Calibrate confidence
        if calibrator and calibrator.is_fitted:
            calibrated_confidence = calibrator.calibrate(raw_confidence)
        else:
            calibrated_confidence = raw_confidence
        
        # Compute prediction entropy (uncertainty measure)
        if prediction_probs is not None:
            # Avoid log(0) by adding small epsilon
            probs = np.clip(prediction_probs, 1e-8, 1.0)
            entropy = -np.sum(probs * np.log(probs))
        else:
            # For regression tasks, use confidence-based entropy approximation
            entropy = -raw_confidence * np.log(raw_confidence + 1e-8) - (1 - raw_confidence) * np.log(1 - raw_confidence + 1e-8)
        
        # Estimate uncertainty (higher entropy = higher uncertainty)
        uncertainty_estimate = entropy / np.log(len(prediction_probs) if prediction_probs is not None else 2)
        
        # Compute confidence interval (approximate)
        margin = 1.96 * np.sqrt(uncertainty_estimate)  # 95% confidence interval
        confidence_interval = (
            max(0.0, calibrated_confidence - margin),
            min(1.0, calibrated_confidence + margin)
        )
        
        # Reliability score (how much to trust this confidence)
        reliability_score = self._compute_reliability_score(raw_confidence, calibrated_confidence, uncertainty_estimate)
        
        return ConfidenceMetrics(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            uncertainty_estimate=uncertainty_estimate,
            prediction_entropy=entropy,
            confidence_interval=confidence_interval,
            reliability_score=reliability_score
        )
    
    def _compute_reliability_score(self, raw_conf: float, cal_conf: float, uncertainty: float) -> float:
        """Compute reliability score for confidence estimate."""
        # Factors that affect reliability:
        # 1. Difference between raw and calibrated confidence (large diff = less reliable)
        # 2. Uncertainty level (high uncertainty = less reliable)
        # 3. Extreme confidence values (very high/low may be overconfident)
        
        calibration_penalty = abs(raw_conf - cal_conf)
        uncertainty_penalty = uncertainty
        extreme_penalty = max(0, abs(raw_conf - 0.5) - 0.4) * 2  # Penalty for extreme confidence
        
        reliability = 1.0 - (calibration_penalty + uncertainty_penalty + extreme_penalty) / 3.0
        return max(0.0, min(1.0, reliability))
    
    def analyze_calibration(self, task: str) -> CalibrationResults:
        """Analyze calibration quality for a specific task.
        
        Args:
            task: Task name ("height", "direction", "breaking")
            
        Returns:
            CalibrationResults with detailed calibration analysis
        """
        if task not in self.calibration_history:
            raise ValueError(f"Unknown task: {task}")
        
        confidences = np.array(self.calibration_history[task]["confidences"])
        accuracies = np.array(self.calibration_history[task]["accuracies"])
        
        if len(confidences) < 10:
            logger.warning(f"Insufficient data for calibration analysis: {len(confidences)}")
            return CalibrationResults(
                expected_calibration_error=1.0,
                maximum_calibration_error=1.0,
                average_calibration_error=1.0,
                reliability_diagram_data={},
                calibration_curve_data=(np.array([]), np.array([])),
                is_well_calibrated=False
            )
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidences, n_bins=10
        )
        
        # Compute calibration errors
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0  # Expected Calibration Error
        mce = 0  # Maximum Calibration Error
        ace = 0  # Average Calibration Error
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
                ace += bin_error
        
        ace /= len(bin_lowers)
        
        # Create reliability diagram data
        reliability_data = {
            "bin_boundaries": bin_boundaries,
            "bin_accuracies": fraction_of_positives,
            "bin_confidences": mean_predicted_value,
            "bin_counts": np.histogram(confidences, bins=bin_boundaries)[0]
        }
        
        # Determine if well calibrated (ECE < 0.1 is generally considered good)
        is_well_calibrated = ece < 0.1
        
        return CalibrationResults(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            average_calibration_error=ace,
            reliability_diagram_data=reliability_data,
            calibration_curve_data=(fraction_of_positives, mean_predicted_value),
            is_well_calibrated=is_well_calibrated
        )
    
    def quantify_uncertainty(self, predictions: Dict[str, torch.Tensor], 
                           num_samples: int = 100) -> UncertaintyQuantification:
        """Quantify prediction uncertainty using Monte Carlo methods.
        
        Args:
            predictions: Model predictions dictionary
            num_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            UncertaintyQuantification with detailed uncertainty analysis
        """
        # For now, implement a simplified uncertainty quantification
        # In a full implementation, this would use techniques like:
        # - Monte Carlo Dropout
        # - Deep Ensembles
        # - Bayesian Neural Networks
        
        # Extract prediction statistics
        if "height_confidence" in predictions:
            confidence = predictions["height_confidence"].item()
            prediction_value = predictions["height_meters"].item()
        elif "direction_confidence" in predictions:
            confidence = predictions["direction_confidence"].item()
            prediction_value = predictions["direction_predicted"].item()
        elif "breaking_confidence" in predictions:
            confidence = predictions["breaking_confidence"].item()
            prediction_value = predictions["breaking_predicted"].item()
        else:
            confidence = 0.5
            prediction_value = 0.0
        
        # Estimate uncertainties (simplified approach)
        # In practice, these would be computed using proper uncertainty quantification methods
        aleatoric_uncertainty = (1.0 - confidence) * 0.5  # Data uncertainty
        epistemic_uncertainty = (1.0 - confidence) * 0.3  # Model uncertainty
        total_uncertainty = np.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2)
        
        # Compute confidence interval
        margin = 1.96 * total_uncertainty
        confidence_interval_95 = (
            prediction_value - margin,
            prediction_value + margin
        )
        
        # Prediction variance
        prediction_variance = total_uncertainty**2
        
        return UncertaintyQuantification(
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_interval_95=confidence_interval_95,
            prediction_variance=prediction_variance
        )
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration status and performance.
        
        Returns:
            Dictionary with calibration summary for all tasks
        """
        summary = {}
        
        for task in ["height", "direction", "breaking"]:
            calibrator = getattr(self, f"{task}_calibrator")
            data_count = len(self.calibration_history[task]["confidences"])
            
            task_summary = {
                "calibrator_fitted": calibrator.is_fitted,
                "data_points": data_count,
                "calibration_method": self.calibration_method
            }
            
            # Add calibration analysis if enough data
            if data_count >= 10:
                try:
                    cal_results = self.analyze_calibration(task)
                    task_summary.update({
                        "expected_calibration_error": cal_results.expected_calibration_error,
                        "is_well_calibrated": cal_results.is_well_calibrated,
                        "maximum_calibration_error": cal_results.maximum_calibration_error
                    })
                except Exception as e:
                    logger.warning(f"Failed to analyze calibration for {task}: {e}")
            
            summary[task] = task_summary
        
        return summary
    
    def clear_calibration_history(self) -> None:
        """Clear all calibration history data."""
        for task in self.calibration_history:
            self.calibration_history[task]["confidences"].clear()
            self.calibration_history[task]["accuracies"].clear()
        
        logger.info("Cleared all calibration history")
    
    def save_calibration_data(self, filepath: str) -> None:
        """Save calibration data to file for later analysis."""
        import pickle
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.calibration_history, f)
            logger.info(f"Saved calibration data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save calibration data: {e}")
    
    def load_calibration_data(self, filepath: str) -> None:
        """Load calibration data from file."""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                self.calibration_history = pickle.load(f)
            logger.info(f"Loaded calibration data from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")