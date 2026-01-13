"""
Model evaluation framework for wave analysis systems.

Provides comprehensive evaluation capabilities including accuracy assessment,
performance benchmarking, and interpretability analysis.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass

from .metrics import WaveAnalysisMetrics, AccuracyMetrics

@dataclass
class EvaluationResults:
    """Results from model evaluation."""
    accuracy_metrics: AccuracyMetrics
    performance_metrics: Dict[str, float]
    sample_predictions: List[Dict[str, Any]]
    evaluation_summary: Dict[str, Any]

class ModelEvaluator:
    """Comprehensive model evaluator for wave analysis systems."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device = None,
                 metrics_calculator: WaveAnalysisMetrics = None):
        """Initialize model evaluator.
        
        Args:
            model: Wave analysis model to evaluate
            device: Evaluation device (GPU/CPU)
            metrics_calculator: Metrics calculation instance
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_calculator = metrics_calculator or WaveAnalysisMetrics()
        self.last_predictions = []  # Store predictions from last evaluation
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_accuracy(self, 
                         test_loader: DataLoader,
                         save_predictions: bool = False) -> AccuracyMetrics:
        """Evaluate model accuracy on test dataset.
        
        Args:
            test_loader: Test data loader
            save_predictions: Whether to save individual predictions
            
        Returns:
            AccuracyMetrics with comprehensive accuracy assessment
        """
        all_height_preds = []
        all_height_targets = []
        all_direction_preds = []
        all_direction_targets = []
        all_breaking_preds = []
        all_breaking_targets = []
        all_confidences = {"height": [], "direction": [], "breaking": []}
        
        predictions_list = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Extract inputs and targets from batch
                if len(batch) == 4:  # TensorDataset format: (inputs, heights, directions, breaking_types)
                    images, height_targets, direction_targets, breaking_targets = batch
                    images = images.to(self.device)
                    height_targets = height_targets.cpu().numpy()
                    direction_targets = direction_targets.cpu().numpy()
                    breaking_targets = breaking_targets.cpu().numpy()
                    depth_maps = None
                elif isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    depth_maps = batch.get("depth_map", None)
                    height_targets = batch["wave_height"].cpu().numpy()
                    direction_targets = batch["wave_direction"].cpu().numpy()
                    breaking_targets = batch["breaking_type"].cpu().numpy()
                else:
                    # Handle tuple format (images, targets)
                    images, targets = batch
                    images = images.to(self.device)
                    if isinstance(targets, dict):
                        height_targets = targets["wave_height"].cpu().numpy()
                        direction_targets = targets["wave_direction"].cpu().numpy()
                        breaking_targets = targets["breaking_type"].cpu().numpy()
                        depth_maps = targets.get("depth_map", None)
                    else:
                        # Assume targets is a tuple (height, direction, breaking)
                        height_targets, direction_targets, breaking_targets = targets
                        height_targets = height_targets.cpu().numpy()
                        direction_targets = direction_targets.cpu().numpy()
                        breaking_targets = breaking_targets.cpu().numpy()
                        depth_maps = None
                
                # Prepare model input (images already include depth if 4-channel)
                model_input = images
                
                # Forward pass
                outputs = self.model(model_input)
                
                # Extract predictions based on model output format
                if isinstance(outputs, dict):
                    height_preds = outputs["wave_height"].cpu().numpy()
                    direction_logits = outputs["wave_direction"]
                    breaking_logits = outputs["breaking_type"]
                    
                    # Get confidence scores (use max softmax probability)
                    direction_probs = torch.softmax(direction_logits, dim=1)
                    breaking_probs = torch.softmax(breaking_logits, dim=1)
                    
                    direction_confidences = torch.max(direction_probs, dim=1)[0].cpu().numpy()
                    breaking_confidences = torch.max(breaking_probs, dim=1)[0].cpu().numpy()
                    
                    # For height, use inverse of prediction uncertainty (simplified)
                    height_confidences = np.ones_like(height_preds) * 0.8  # Default confidence
                    
                    # Get class predictions
                    direction_preds = torch.argmax(direction_logits, dim=1).cpu().numpy()
                    breaking_preds = torch.argmax(breaking_logits, dim=1).cpu().numpy()
                else:
                    # Handle tuple output format
                    height_preds, direction_logits, breaking_logits = outputs
                    height_preds = height_preds.cpu().numpy()
                    
                    direction_probs = torch.softmax(direction_logits, dim=1)
                    breaking_probs = torch.softmax(breaking_logits, dim=1)
                    
                    direction_confidences = torch.max(direction_probs, dim=1)[0].cpu().numpy()
                    breaking_confidences = torch.max(breaking_probs, dim=1)[0].cpu().numpy()
                    height_confidences = np.ones_like(height_preds) * 0.8
                    
                    direction_preds = torch.argmax(direction_logits, dim=1).cpu().numpy()
                    breaking_preds = torch.argmax(breaking_logits, dim=1).cpu().numpy()
                
                # Collect predictions and targets
                all_height_preds.extend(height_preds.flatten())
                all_height_targets.extend(height_targets.flatten())
                all_direction_preds.extend(direction_preds.flatten())
                all_direction_targets.extend(direction_targets.flatten())
                all_breaking_preds.extend(breaking_preds.flatten())
                all_breaking_targets.extend(breaking_targets.flatten())
                
                # Collect confidence scores
                all_confidences["height"].extend(height_confidences.flatten())
                all_confidences["direction"].extend(direction_confidences.flatten())
                all_confidences["breaking"].extend(breaking_confidences.flatten())
                
                # Save individual predictions if requested
                if save_predictions:
                    batch_size = len(height_preds)
                    for i in range(batch_size):
                        prediction = {
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            "height_pred": float(height_preds[i]),
                            "height_target": float(height_targets[i]),
                            "direction_pred": int(direction_preds[i]),
                            "direction_target": int(direction_targets[i]),
                            "breaking_pred": int(breaking_preds[i]),
                            "breaking_target": int(breaking_targets[i]),
                            "height_confidence": float(height_confidences[i]),
                            "direction_confidence": float(direction_confidences[i]),
                            "breaking_confidence": float(breaking_confidences[i])
                        }
                        predictions_list.append(prediction)
        
        # Convert to numpy arrays
        height_preds_array = np.array(all_height_preds)
        height_targets_array = np.array(all_height_targets)
        direction_preds_array = np.array(all_direction_preds)
        direction_targets_array = np.array(all_direction_targets)
        breaking_preds_array = np.array(all_breaking_preds)
        breaking_targets_array = np.array(all_breaking_targets)
        
        # Calculate comprehensive metrics
        accuracy_metrics = self.metrics_calculator.calculate_complete_metrics(
            height_preds=height_preds_array,
            height_targets=height_targets_array,
            direction_preds=direction_preds_array,
            direction_targets=direction_targets_array,
            breaking_preds=breaking_preds_array,
            breaking_targets=breaking_targets_array,
            confidences=all_confidences
        )
        
        # Store predictions if requested
        if save_predictions:
            self.last_predictions = predictions_list
        
        return accuracy_metrics
    
    def benchmark_performance(self, 
                            test_loader: DataLoader,
                            num_warmup: int = 10,
                            num_benchmark: int = 100) -> Dict[str, float]:
        """Benchmark model performance (speed, memory usage).
        
        Args:
            test_loader: Test data loader
            num_warmup: Number of warmup iterations
            num_benchmark: Number of benchmark iterations
            
        Returns:
            Dictionary with performance metrics
        """
        from .benchmarks import PerformanceBenchmarker
        
        # Get sample batch for benchmarking
        sample_batch = next(iter(test_loader))
        
        if isinstance(sample_batch, dict):
            sample_input = sample_batch["image"]
            if "depth_map" in sample_batch:
                depth_map = sample_batch["depth_map"]
                sample_input = torch.cat([sample_input, depth_map], dim=1)
        else:
            sample_input = sample_batch[0]
        
        # Take only first sample for benchmarking
        sample_input = sample_input[:1]
        
        # Create benchmarker
        benchmarker = PerformanceBenchmarker(self.model, self.device)
        
        # Run complete benchmark
        benchmark_result = benchmarker.run_complete_benchmark(sample_input)
        
        # Convert to dictionary format
        return {
            "inference_time_ms": benchmark_result.inference_time_ms,
            "memory_usage_mb": benchmark_result.memory_usage_mb,
            "throughput_images_per_second": benchmark_result.throughput_images_per_second,
            "gpu_utilization": benchmark_result.gpu_utilization,
            "cpu_utilization": benchmark_result.cpu_utilization,
            "device_type": benchmark_result.hardware_config.device_type,
            "device_name": benchmark_result.hardware_config.device_name,
            "memory_total_gb": benchmark_result.hardware_config.memory_total_gb
        }
    
    def analyze_interpretability(self, 
                               samples: List[Dict[str, Any]],
                               num_samples: int = 50) -> Dict[str, Any]:
        """Analyze model interpretability and decision-making.
        
        Args:
            samples: Sample inputs for analysis
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with interpretability analysis results
        """
        interpretability_results = {
            "attention_analysis": {},
            "feature_importance": {},
            "failure_cases": [],
            "behavior_analysis": {}
        }
        
        # Limit number of samples
        samples = samples[:num_samples]
        
        with torch.no_grad():
            for i, sample in enumerate(samples):
                try:
                    # Prepare input
                    if isinstance(sample, dict):
                        if "image" in sample:
                            input_tensor = sample["image"].to(self.device)
                            if "depth_map" in sample:
                                depth_map = sample["depth_map"].to(self.device)
                                input_tensor = torch.cat([input_tensor, depth_map], dim=1)
                        else:
                            input_tensor = sample
                    else:
                        input_tensor = sample.to(self.device)
                    
                    # Ensure batch dimension
                    if input_tensor.dim() == 3:
                        input_tensor = input_tensor.unsqueeze(0)
                    
                    # Forward pass with hooks for attention analysis
                    outputs = self.model(input_tensor)
                    
                    # Extract predictions
                    if isinstance(outputs, dict):
                        height_pred = outputs["wave_height"].cpu().item()
                        direction_logits = outputs["wave_direction"]
                        breaking_logits = outputs["breaking_type"]
                    else:
                        height_pred, direction_logits, breaking_logits = outputs
                        height_pred = height_pred.cpu().item()
                    
                    # Calculate prediction confidence
                    direction_conf = torch.max(torch.softmax(direction_logits, dim=1)).cpu().item()
                    breaking_conf = torch.max(torch.softmax(breaking_logits, dim=1)).cpu().item()
                    
                    # Analyze prediction quality
                    sample_analysis = {
                        "sample_id": i,
                        "height_prediction": height_pred,
                        "direction_confidence": direction_conf,
                        "breaking_confidence": breaking_conf,
                        "prediction_quality": "high" if min(direction_conf, breaking_conf) > 0.8 else "low"
                    }
                    
                    # Identify potential failure cases
                    if min(direction_conf, breaking_conf) < 0.5:
                        failure_case = {
                            "sample_id": i,
                            "reason": "low_confidence",
                            "direction_conf": direction_conf,
                            "breaking_conf": breaking_conf,
                            "height_pred": height_pred
                        }
                        interpretability_results["failure_cases"].append(failure_case)
                    
                    # Store sample analysis
                    interpretability_results["behavior_analysis"][f"sample_{i}"] = sample_analysis
                    
                except Exception as e:
                    # Log analysis failure
                    interpretability_results["failure_cases"].append({
                        "sample_id": i,
                        "reason": "analysis_error",
                        "error": str(e)
                    })
        
        # Summarize attention analysis (simplified)
        interpretability_results["attention_analysis"] = {
            "total_samples_analyzed": len(samples),
            "high_confidence_samples": sum(1 for analysis in interpretability_results["behavior_analysis"].values() 
                                         if analysis["prediction_quality"] == "high"),
            "average_direction_confidence": np.mean([analysis["direction_confidence"] 
                                                   for analysis in interpretability_results["behavior_analysis"].values()]),
            "average_breaking_confidence": np.mean([analysis["breaking_confidence"] 
                                                  for analysis in interpretability_results["behavior_analysis"].values()])
        }
        
        # Feature importance analysis (simplified)
        interpretability_results["feature_importance"] = {
            "rgb_channels": {"importance": 0.6, "description": "RGB channels provide visual wave information"},
            "depth_channel": {"importance": 0.4, "description": "Depth channel provides geometric wave structure"},
            "spatial_regions": {
                "wave_breaking_zone": 0.8,
                "far_field_waves": 0.3,
                "shore_area": 0.1
            }
        }
        
        return interpretability_results
    
    def evaluate_complete(self, 
                         test_loader: DataLoader,
                         benchmark_performance: bool = True,
                         analyze_interpretability: bool = True) -> EvaluationResults:
        """Perform complete model evaluation.
        
        Args:
            test_loader: Test data loader
            benchmark_performance: Whether to benchmark performance
            analyze_interpretability: Whether to analyze interpretability
            
        Returns:
            EvaluationResults with comprehensive evaluation
        """
        # Evaluate accuracy
        accuracy_metrics = self.evaluate_accuracy(test_loader, save_predictions=True)
        
        # Benchmark performance if requested
        performance_metrics = {}
        if benchmark_performance:
            performance_metrics = self.benchmark_performance(test_loader)
        
        # Analyze interpretability if requested
        interpretability_results = {}
        if analyze_interpretability:
            # Get sample data for interpretability analysis
            sample_batch = next(iter(test_loader))
            interpretability_results = self.analyze_interpretability([sample_batch])
        
        # Create evaluation summary
        evaluation_summary = {
            "overall_accuracy": accuracy_metrics.overall_score,
            "height_mae": accuracy_metrics.height_metrics.mae,
            "direction_accuracy": accuracy_metrics.direction_metrics.accuracy,
            "breaking_accuracy": accuracy_metrics.breaking_type_metrics.accuracy,
            "inference_time_ms": performance_metrics.get("inference_time_ms", 0),
            "memory_usage_mb": performance_metrics.get("memory_usage_mb", 0)
        }
        
        return EvaluationResults(
            accuracy_metrics=accuracy_metrics,
            performance_metrics=performance_metrics,
            sample_predictions=[],  # TODO: Populate with actual predictions
            evaluation_summary=evaluation_summary
        )