"""
Evaluation and metrics for the SwellSight Wave Analysis System.

This module contains evaluation frameworks, wave-specific metrics, benchmarking tools,
and comprehensive reporting for model performance assessment.
"""

from .metrics import WaveAnalysisMetrics
from .evaluator import ModelEvaluator
from .benchmarks import PerformanceBenchmark
from .reports import EvaluationReporter

__all__ = [
    "WaveAnalysisMetrics",
    "ModelEvaluator",
    "PerformanceBenchmark", 
    "EvaluationReporter"
]