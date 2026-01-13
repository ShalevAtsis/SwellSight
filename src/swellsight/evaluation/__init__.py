"""
Evaluation and metrics for the SwellSight Wave Analysis System.

This module contains evaluation frameworks, wave-specific metrics, benchmarking tools,
and comprehensive reporting for model performance assessment.
"""

# Import the new data evaluation modules (these don't have circular dependencies)
from .data_quality import DataQualityAssessor, QualityReport
from .data_comparison import DatasetComparator, DistributionComparison, DataDriftMetrics
from .data_insights import DataInsightsReporter, DataLineageTracker, DataVersionManager, DataHealthMonitor

__all__ = [
    "DataQualityAssessor",
    "QualityReport",
    "DatasetComparator",
    "DistributionComparison",
    "DataDriftMetrics",
    "DataInsightsReporter",
    "DataLineageTracker",
    "DataVersionManager",
    "DataHealthMonitor"
]