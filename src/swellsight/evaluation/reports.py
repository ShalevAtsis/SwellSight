"""
Evaluation reporting system for wave analysis models.

Creates comprehensive evaluation reports with metrics, visualizations,
and recommendations for model performance assessment.
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import AccuracyMetrics
from .benchmarks import PerformanceBenchmark

@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    report_id: str
    timestamp: datetime
    model_info: Dict[str, Any]
    accuracy_metrics: AccuracyMetrics
    performance_benchmark: PerformanceBenchmark
    visualizations: Dict[str, str]  # Paths to generated plots
    recommendations: List[str]
    summary: Dict[str, Any]

class EvaluationReporter:
    """Evaluation report generator for wave analysis models."""
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        """Initialize evaluation reporter.
        
        Args:
            output_dir: Directory to save evaluation reports
        """
        self.output_dir = output_dir
        
    def generate_evaluation_report(self,
                                 model_info: Dict[str, Any],
                                 accuracy_metrics: AccuracyMetrics,
                                 performance_benchmark: PerformanceBenchmark,
                                 additional_data: Optional[Dict[str, Any]] = None) -> EvaluationReport:
        """Generate comprehensive evaluation report.
        
        Args:
            model_info: Information about the evaluated model
            accuracy_metrics: Accuracy evaluation results
            performance_benchmark: Performance benchmark results
            additional_data: Additional evaluation data
            
        Returns:
            EvaluationReport with comprehensive evaluation results
        """
        import os
        from pathlib import Path
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate unique report ID
        timestamp = datetime.now()
        report_id = f"eval_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create visualizations
        visualizations = self.create_visualizations(accuracy_metrics, performance_benchmark)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(accuracy_metrics, performance_benchmark)
        
        # Create summary
        summary = {
            "overall_score": accuracy_metrics.overall_score,
            "height_accuracy": {
                "mae": accuracy_metrics.height_metrics.mae,
                "rmse": accuracy_metrics.height_metrics.rmse,
                "accuracy_02m": accuracy_metrics.height_metrics.accuracy_within_02m
            },
            "classification_accuracy": {
                "direction": accuracy_metrics.direction_metrics.accuracy,
                "breaking_type": accuracy_metrics.breaking_type_metrics.accuracy
            },
            "performance": {
                "inference_time_ms": performance_benchmark.inference_time_ms,
                "throughput_fps": performance_benchmark.throughput_images_per_second,
                "memory_usage_mb": performance_benchmark.memory_usage_mb
            },
            "calibration": {
                "expected_calibration_error": accuracy_metrics.confidence_calibration.expected_calibration_error,
                "maximum_calibration_error": accuracy_metrics.confidence_calibration.maximum_calibration_error
            }
        }
        
        return EvaluationReport(
            report_id=report_id,
            timestamp=timestamp,
            model_info=model_info,
            accuracy_metrics=accuracy_metrics,
            performance_benchmark=performance_benchmark,
            visualizations=visualizations,
            recommendations=recommendations,
            summary=summary
        )
    
    def create_visualizations(self,
                            accuracy_metrics: AccuracyMetrics,
                            performance_benchmark: PerformanceBenchmark) -> Dict[str, str]:
        """Create evaluation visualizations.
        
        Args:
            accuracy_metrics: Accuracy evaluation results
            performance_benchmark: Performance benchmark results
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        import os
        from pathlib import Path
        
        visualizations = {}
        viz_dir = Path(self.output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion matrices for classification tasks
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Direction confusion matrix
        sns.heatmap(accuracy_metrics.direction_metrics.confusion_matrix, 
                   annot=True, fmt='d', ax=axes[0],
                   xticklabels=['LEFT', 'RIGHT', 'STRAIGHT'],
                   yticklabels=['LEFT', 'RIGHT', 'STRAIGHT'])
        axes[0].set_title('Wave Direction Classification')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Breaking type confusion matrix
        sns.heatmap(accuracy_metrics.breaking_type_metrics.confusion_matrix, 
                   annot=True, fmt='d', ax=axes[1],
                   xticklabels=['SPILLING', 'PLUNGING', 'SURGING'],
                   yticklabels=['SPILLING', 'PLUNGING', 'SURGING'])
        axes[1].set_title('Breaking Type Classification')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        confusion_path = viz_dir / "confusion_matrices.png"
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations["confusion_matrices"] = str(confusion_path)
        
        # 2. Reliability diagram for calibration
        if accuracy_metrics.confidence_calibration.reliability_diagram_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            rel_data = accuracy_metrics.confidence_calibration.reliability_diagram_data
            if rel_data.get("bin_centers") and rel_data.get("bin_accuracies"):
                bin_centers = rel_data["bin_centers"]
                bin_accuracies = rel_data["bin_accuracies"]
                bin_confidences = rel_data["bin_confidences"]
                
                # Plot reliability diagram
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                ax.scatter(bin_confidences, bin_accuracies, s=100, alpha=0.7, label='Model Calibration')
                
                ax.set_xlabel('Mean Predicted Confidence')
                ax.set_ylabel('Accuracy')
                ax.set_title('Confidence Calibration (Reliability Diagram)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                reliability_path = viz_dir / "reliability_diagram.png"
                plt.savefig(reliability_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations["reliability_diagram"] = str(reliability_path)
        
        # 3. Performance metrics visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Height metrics
        height_metrics = ['MAE', 'RMSE', 'Acc@0.2m', 'Acc@0.5m']
        height_values = [
            accuracy_metrics.height_metrics.mae,
            accuracy_metrics.height_metrics.rmse,
            accuracy_metrics.height_metrics.accuracy_within_02m,
            accuracy_metrics.height_metrics.accuracy_within_05m
        ]
        axes[0, 0].bar(height_metrics, height_values)
        axes[0, 0].set_title('Wave Height Prediction Metrics')
        axes[0, 0].set_ylabel('Value')
        
        # Classification accuracies
        class_tasks = ['Direction', 'Breaking Type']
        class_accuracies = [
            accuracy_metrics.direction_metrics.accuracy * 100,
            accuracy_metrics.breaking_type_metrics.accuracy * 100
        ]
        axes[0, 1].bar(class_tasks, class_accuracies)
        axes[0, 1].set_title('Classification Accuracies (%)')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_ylim(0, 100)
        
        # Performance metrics
        perf_metrics = ['Inference Time (ms)', 'Memory Usage (MB)', 'Throughput (FPS)']
        perf_values = [
            performance_benchmark.inference_time_ms,
            performance_benchmark.memory_usage_mb,
            performance_benchmark.throughput_images_per_second
        ]
        axes[1, 0].bar(perf_metrics, perf_values)
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Value')
        
        # Hardware utilization
        util_metrics = ['GPU Util (%)', 'CPU Util (%)']
        util_values = [
            performance_benchmark.gpu_utilization,
            performance_benchmark.cpu_utilization
        ]
        axes[1, 1].bar(util_metrics, util_values)
        axes[1, 1].set_title('Hardware Utilization')
        axes[1, 1].set_ylabel('Utilization (%)')
        axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        metrics_path = viz_dir / "performance_metrics.png"
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations["performance_metrics"] = str(metrics_path)
        
        return visualizations
    
    def generate_recommendations(self,
                               accuracy_metrics: AccuracyMetrics,
                               performance_benchmark: PerformanceBenchmark) -> List[str]:
        """Generate recommendations based on evaluation results.
        
        Args:
            accuracy_metrics: Accuracy evaluation results
            performance_benchmark: Performance benchmark results
            
        Returns:
            List of recommendations for model improvement
        """
        recommendations = []
        
        # Height prediction recommendations
        if accuracy_metrics.height_metrics.mae > 0.3:
            recommendations.append(
                f"Wave height MAE ({accuracy_metrics.height_metrics.mae:.3f}m) is above target. "
                "Consider increasing training data diversity or adjusting loss weighting."
            )
        
        if accuracy_metrics.height_metrics.accuracy_within_02m < 80:
            recommendations.append(
                f"Height accuracy within ±0.2m ({accuracy_metrics.height_metrics.accuracy_within_02m:.1f}%) "
                "is below target (80%). Focus on improving precision for small wave measurements."
            )
        
        # Direction classification recommendations
        if accuracy_metrics.direction_metrics.accuracy < 0.9:
            recommendations.append(
                f"Direction classification accuracy ({accuracy_metrics.direction_metrics.accuracy:.3f}) "
                "is below target (90%). Consider augmenting training data with more directional variations."
            )
        
        # Breaking type classification recommendations
        if accuracy_metrics.breaking_type_metrics.accuracy < 0.92:
            recommendations.append(
                f"Breaking type accuracy ({accuracy_metrics.breaking_type_metrics.accuracy:.3f}) "
                "is below target (92%). Focus on collecting more diverse breaking type examples."
            )
        
        # Calibration recommendations
        if accuracy_metrics.confidence_calibration.expected_calibration_error > 0.1:
            recommendations.append(
                f"Expected Calibration Error ({accuracy_metrics.confidence_calibration.expected_calibration_error:.3f}) "
                "is high. Consider temperature scaling or Platt scaling for better confidence calibration."
            )
        
        # Performance recommendations
        if performance_benchmark.inference_time_ms > 200:
            recommendations.append(
                f"Inference time ({performance_benchmark.inference_time_ms:.1f}ms) exceeds target (200ms). "
                "Consider model optimization techniques like quantization or pruning."
            )
        
        if performance_benchmark.memory_usage_mb > 2000:
            recommendations.append(
                f"Memory usage ({performance_benchmark.memory_usage_mb:.1f}MB) is high. "
                "Consider reducing model size or using gradient checkpointing."
            )
        
        if performance_benchmark.throughput_images_per_second < 2:
            recommendations.append(
                f"Throughput ({performance_benchmark.throughput_images_per_second:.1f} FPS) is below target (2 FPS). "
                "Optimize model architecture or consider batch processing."
            )
        
        # Hardware utilization recommendations
        if performance_benchmark.gpu_utilization < 50 and performance_benchmark.hardware_config.device_type == "cuda":
            recommendations.append(
                f"GPU utilization ({performance_benchmark.gpu_utilization:.1f}%) is low. "
                "Consider increasing batch size or using mixed precision training."
            )
        
        # Overall performance recommendations
        if accuracy_metrics.overall_score < 80:
            recommendations.append(
                f"Overall model score ({accuracy_metrics.overall_score:.1f}%) needs improvement. "
                "Focus on the lowest-performing task and consider multi-task learning adjustments."
            )
        
        # Add positive feedback if performance is good
        if not recommendations:
            recommendations.append(
                "Model performance meets all targets. Consider deployment or further optimization for production."
            )
        
        return recommendations
    
    def save_report(self, report: EvaluationReport, format: str = "json") -> str:
        """Save evaluation report to file.
        
        Args:
            report: Evaluation report to save
            format: Output format ("json", "html", "pdf")
            
        Returns:
            Path to saved report file
        """
        from pathlib import Path
        
        output_path = Path(self.output_dir) / f"{report.report_id}.{format}"
        
        if format == "json":
            # Convert report to JSON-serializable format
            report_dict = {
                "report_id": report.report_id,
                "timestamp": report.timestamp.isoformat(),
                "model_info": report.model_info,
                "accuracy_metrics": {
                    "height_metrics": asdict(report.accuracy_metrics.height_metrics),
                    "direction_metrics": {
                        "accuracy": report.accuracy_metrics.direction_metrics.accuracy,
                        "precision_per_class": report.accuracy_metrics.direction_metrics.precision_per_class,
                        "recall_per_class": report.accuracy_metrics.direction_metrics.recall_per_class,
                        "f1_score_per_class": report.accuracy_metrics.direction_metrics.f1_score_per_class,
                        "confusion_matrix": report.accuracy_metrics.direction_metrics.confusion_matrix.tolist(),
                        "macro_avg_f1": report.accuracy_metrics.direction_metrics.macro_avg_f1
                    },
                    "breaking_type_metrics": {
                        "accuracy": report.accuracy_metrics.breaking_type_metrics.accuracy,
                        "precision_per_class": report.accuracy_metrics.breaking_type_metrics.precision_per_class,
                        "recall_per_class": report.accuracy_metrics.breaking_type_metrics.recall_per_class,
                        "f1_score_per_class": report.accuracy_metrics.breaking_type_metrics.f1_score_per_class,
                        "confusion_matrix": report.accuracy_metrics.breaking_type_metrics.confusion_matrix.tolist(),
                        "macro_avg_f1": report.accuracy_metrics.breaking_type_metrics.macro_avg_f1
                    },
                    "confidence_calibration": asdict(report.accuracy_metrics.confidence_calibration),
                    "overall_score": report.accuracy_metrics.overall_score
                },
                "performance_benchmark": asdict(report.performance_benchmark),
                "visualizations": report.visualizations,
                "recommendations": report.recommendations,
                "summary": report.summary
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
        
        elif format == "html":
            # Generate HTML report
            html_content = self._generate_html_report(report)
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_path)
    
    def _generate_html_report(self, report: EvaluationReport) -> str:
        """Generate HTML report content."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wave Analysis Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Wave Analysis Model Evaluation Report</h1>
                <p><strong>Report ID:</strong> {report.report_id}</p>
                <p><strong>Generated:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Overall Score:</strong> {report.accuracy_metrics.overall_score:.2f}%</p>
            </div>
            
            <div class="section">
                <h2>Model Information</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in report.model_info.items())}
                </table>
            </div>
            
            <div class="section">
                <h2>Accuracy Metrics</h2>
                <div class="metric">
                    <h3>Wave Height Prediction</h3>
                    <p>MAE: {report.accuracy_metrics.height_metrics.mae:.3f}m</p>
                    <p>RMSE: {report.accuracy_metrics.height_metrics.rmse:.3f}m</p>
                    <p>Accuracy within ±0.2m: {report.accuracy_metrics.height_metrics.accuracy_within_02m:.1f}%</p>
                </div>
                <div class="metric">
                    <h3>Direction Classification</h3>
                    <p>Accuracy: {report.accuracy_metrics.direction_metrics.accuracy:.3f}</p>
                    <p>Macro F1: {report.accuracy_metrics.direction_metrics.macro_avg_f1:.3f}</p>
                </div>
                <div class="metric">
                    <h3>Breaking Type Classification</h3>
                    <p>Accuracy: {report.accuracy_metrics.breaking_type_metrics.accuracy:.3f}</p>
                    <p>Macro F1: {report.accuracy_metrics.breaking_type_metrics.macro_avg_f1:.3f}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="metric">
                    <p>Inference Time: {report.performance_benchmark.inference_time_ms:.1f}ms</p>
                    <p>Memory Usage: {report.performance_benchmark.memory_usage_mb:.1f}MB</p>
                    <p>Throughput: {report.performance_benchmark.throughput_images_per_second:.1f} FPS</p>
                    <p>Device: {report.performance_benchmark.hardware_config.device_name}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {''.join(f'<div class="recommendation">{rec}</div>' for rec in report.recommendations)}
            </div>
        </body>
        </html>
        """
        return html_template
    
    def compare_models(self, reports: List[EvaluationReport]) -> Dict[str, Any]:
        """Compare multiple model evaluation reports.
        
        Args:
            reports: List of evaluation reports to compare
            
        Returns:
            Comparison analysis results
        """
        if len(reports) < 2:
            return {"error": "At least 2 reports required for comparison"}
        
        comparison = {
            "num_models": len(reports),
            "comparison_timestamp": datetime.now().isoformat(),
            "models": [],
            "best_performers": {},
            "performance_trends": {}
        }
        
        # Extract metrics for each model
        for i, report in enumerate(reports):
            model_summary = {
                "model_id": report.report_id,
                "timestamp": report.timestamp.isoformat(),
                "overall_score": report.accuracy_metrics.overall_score,
                "height_mae": report.accuracy_metrics.height_metrics.mae,
                "direction_accuracy": report.accuracy_metrics.direction_metrics.accuracy,
                "breaking_accuracy": report.accuracy_metrics.breaking_type_metrics.accuracy,
                "inference_time_ms": report.performance_benchmark.inference_time_ms,
                "memory_usage_mb": report.performance_benchmark.memory_usage_mb,
                "throughput_fps": report.performance_benchmark.throughput_images_per_second
            }
            comparison["models"].append(model_summary)
        
        # Find best performers for each metric
        models = comparison["models"]
        
        comparison["best_performers"] = {
            "highest_overall_score": max(models, key=lambda x: x["overall_score"]),
            "lowest_height_mae": min(models, key=lambda x: x["height_mae"]),
            "highest_direction_accuracy": max(models, key=lambda x: x["direction_accuracy"]),
            "highest_breaking_accuracy": max(models, key=lambda x: x["breaking_accuracy"]),
            "fastest_inference": min(models, key=lambda x: x["inference_time_ms"]),
            "lowest_memory_usage": min(models, key=lambda x: x["memory_usage_mb"]),
            "highest_throughput": max(models, key=lambda x: x["throughput_fps"])
        }
        
        # Calculate performance trends (if models are chronologically ordered)
        if len(models) > 1:
            first_model = models[0]
            last_model = models[-1]
            
            comparison["performance_trends"] = {
                "overall_score_change": last_model["overall_score"] - first_model["overall_score"],
                "height_mae_change": last_model["height_mae"] - first_model["height_mae"],
                "direction_accuracy_change": last_model["direction_accuracy"] - first_model["direction_accuracy"],
                "breaking_accuracy_change": last_model["breaking_accuracy"] - first_model["breaking_accuracy"],
                "inference_time_change": last_model["inference_time_ms"] - first_model["inference_time_ms"],
                "memory_usage_change": last_model["memory_usage_mb"] - first_model["memory_usage_mb"]
            }
        
        return comparison