"""
Data Insights and Reporting System for SwellSight Wave Analysis System.

This module provides automated data quality reports, data lineage tracking,
metadata management, data versioning, and data health monitoring dashboards.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import hashlib
import sqlite3
from contextlib import contextmanager

from .data_quality import DataQualityAssessor, QualityReport
from .data_comparison import DatasetComparator, DistributionComparison, DataDriftMetrics

logger = logging.getLogger(__name__)


@dataclass
class DataLineage:
    """Tracks the lineage and provenance of data."""
    dataset_id: str
    source_path: str
    creation_timestamp: datetime
    processing_steps: List[Dict[str, Any]]
    parent_datasets: List[str]
    metadata: Dict[str, Any]
    checksum: str
    version: str


@dataclass
class DataVersion:
    """Represents a version of a dataset."""
    version_id: str
    dataset_name: str
    version_number: str
    creation_date: datetime
    description: str
    changes: List[str]
    metrics: Dict[str, float]
    file_path: str
    size_bytes: int
    checksum: str


@dataclass
class ExperimentMetadata:
    """Metadata for machine learning experiments."""
    experiment_id: str
    experiment_name: str
    dataset_versions: List[str]
    model_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # 'running', 'completed', 'failed'
    notes: str


@dataclass
class DataHealthMetrics:
    """Metrics for monitoring data health over time."""
    timestamp: datetime
    dataset_name: str
    total_samples: int
    quality_score: float
    drift_score: float
    anomaly_count: int
    missing_data_percentage: float
    duplicate_percentage: float
    format_consistency_score: float
    freshness_hours: float
    alerts: List[str]


@dataclass
class DataInsightReport:
    """Comprehensive data insights report."""
    report_id: str
    generation_timestamp: datetime
    dataset_summary: Dict[str, Any]
    quality_analysis: QualityReport
    distribution_analysis: Optional[DistributionComparison]
    drift_analysis: Optional[DataDriftMetrics]
    lineage_info: DataLineage
    health_metrics: DataHealthMetrics
    recommendations: List[str]
    visualizations: Dict[str, str]  # visualization_name -> file_path


class DataLineageTracker:
    """
    Tracks data lineage and provenance throughout the pipeline.
    """
    
    def __init__(self, db_path: str = "data_lineage.db"):
        """Initialize the lineage tracker with SQLite database."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for lineage tracking."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    creation_timestamp TEXT NOT NULL,
                    processing_steps TEXT,
                    parent_datasets TEXT,
                    metadata TEXT,
                    checksum TEXT,
                    version TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_steps (
                    step_id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    step_name TEXT,
                    step_config TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def register_dataset(self, 
                        dataset_path: Union[str, Path],
                        parent_datasets: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new dataset in the lineage system.
        
        Args:
            dataset_path: Path to the dataset
            parent_datasets: List of parent dataset IDs
            metadata: Additional metadata
            
        Returns:
            str: Dataset ID
        """
        dataset_path = Path(dataset_path)
        
        # Generate dataset ID
        dataset_id = self._generate_dataset_id(dataset_path)
        
        # Calculate checksum
        checksum = self._calculate_dataset_checksum(dataset_path)
        
        # Create lineage record
        lineage = DataLineage(
            dataset_id=dataset_id,
            source_path=str(dataset_path),
            creation_timestamp=datetime.now(),
            processing_steps=[],
            parent_datasets=parent_datasets or [],
            metadata=metadata or {},
            checksum=checksum,
            version="1.0"
        )
        
        # Store in database
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO datasets 
                (dataset_id, source_path, creation_timestamp, processing_steps, 
                 parent_datasets, metadata, checksum, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lineage.dataset_id,
                lineage.source_path,
                lineage.creation_timestamp.isoformat(),
                json.dumps(lineage.processing_steps),
                json.dumps(lineage.parent_datasets),
                json.dumps(lineage.metadata),
                lineage.checksum,
                lineage.version
            ))
        
        logger.info(f"Registered dataset {dataset_id} in lineage system")
        return dataset_id
    
    def add_processing_step(self, 
                           dataset_id: str,
                           step_name: str,
                           step_config: Dict[str, Any]) -> None:
        """
        Add a processing step to the dataset lineage.
        
        Args:
            dataset_id: Dataset ID
            step_name: Name of the processing step
            step_config: Configuration used for the step
        """
        step_id = f"{dataset_id}_{step_name}_{datetime.now().timestamp()}"
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO processing_steps 
                (step_id, dataset_id, step_name, step_config, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                step_id,
                dataset_id,
                step_name,
                json.dumps(step_config),
                datetime.now().isoformat()
            ))
        
        logger.info(f"Added processing step '{step_name}' to dataset {dataset_id}")
    
    def get_lineage(self, dataset_id: str) -> Optional[DataLineage]:
        """
        Get the complete lineage for a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            DataLineage: Complete lineage information
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM datasets WHERE dataset_id = ?
            """, (dataset_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get processing steps
            steps_cursor = conn.execute("""
                SELECT step_name, step_config, timestamp 
                FROM processing_steps 
                WHERE dataset_id = ?
                ORDER BY timestamp
            """, (dataset_id,))
            
            processing_steps = []
            for step_row in steps_cursor.fetchall():
                processing_steps.append({
                    'step_name': step_row[0],
                    'step_config': json.loads(step_row[1]),
                    'timestamp': step_row[2]
                })
            
            return DataLineage(
                dataset_id=row[0],
                source_path=row[1],
                creation_timestamp=datetime.fromisoformat(row[2]),
                processing_steps=processing_steps,
                parent_datasets=json.loads(row[4]),
                metadata=json.loads(row[5]),
                checksum=row[6],
                version=row[7]
            )
    
    def _generate_dataset_id(self, dataset_path: Path) -> str:
        """Generate a unique dataset ID."""
        path_str = str(dataset_path.absolute())
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{path_str}_{timestamp}".encode()).hexdigest()[:16]
    
    def _calculate_dataset_checksum(self, dataset_path: Path) -> str:
        """Calculate checksum for dataset integrity."""
        if dataset_path.is_file():
            with open(dataset_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        else:
            # For directories, hash the list of files and their sizes
            files_info = []
            for file_path in sorted(dataset_path.rglob('*')):
                if file_path.is_file():
                    files_info.append(f"{file_path.name}:{file_path.stat().st_size}")
            
            combined = "|".join(files_info)
            return hashlib.md5(combined.encode()).hexdigest()


class DataVersionManager:
    """
    Manages data versioning and tracks changes over time.
    """
    
    def __init__(self, versions_dir: str = "data_versions"):
        """Initialize the version manager."""
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        self.metadata_file = self.versions_dir / "versions_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing version metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"versions": {}, "datasets": {}}
    
    def _save_metadata(self):
        """Save version metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def create_version(self, 
                      dataset_name: str,
                      dataset_path: Union[str, Path],
                      description: str,
                      changes: List[str]) -> str:
        """
        Create a new version of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset
            description: Description of this version
            changes: List of changes made
            
        Returns:
            str: Version ID
        """
        dataset_path = Path(dataset_path)
        
        # Generate version number
        if dataset_name not in self.metadata["datasets"]:
            self.metadata["datasets"][dataset_name] = {"versions": [], "latest": None}
            version_number = "1.0.0"
        else:
            latest_version = self.metadata["datasets"][dataset_name]["latest"]
            if latest_version:
                major, minor, patch = map(int, latest_version.split('.'))
                version_number = f"{major}.{minor}.{patch + 1}"
            else:
                version_number = "1.0.0"
        
        # Generate version ID
        version_id = f"{dataset_name}_v{version_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate metrics
        try:
            assessor = DataQualityAssessor()
            quality_report = assessor.assess_quality(dataset_path)
            metrics = {
                "quality_score": quality_report.overall_score,
                "total_images": quality_report.dataset_statistics.total_images,
                "total_size_mb": quality_report.dataset_statistics.total_size_mb
            }
        except Exception as e:
            logger.warning(f"Could not calculate metrics for version: {e}")
            metrics = {}
        
        # Create version record
        version = DataVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            version_number=version_number,
            creation_date=datetime.now(),
            description=description,
            changes=changes,
            metrics=metrics,
            file_path=str(dataset_path),
            size_bytes=self._calculate_size(dataset_path),
            checksum=self._calculate_checksum(dataset_path)
        )
        
        # Update metadata
        self.metadata["versions"][version_id] = {
            "dataset_name": dataset_name,
            "version_number": version_number,
            "creation_date": version.creation_date.isoformat(),
            "description": description,
            "changes": changes,
            "metrics": metrics,
            "file_path": str(dataset_path),
            "size_bytes": version.size_bytes,
            "checksum": version.checksum
        }
        
        self.metadata["datasets"][dataset_name]["versions"].append(version_id)
        self.metadata["datasets"][dataset_name]["latest"] = version_number
        
        self._save_metadata()
        
        logger.info(f"Created version {version_number} for dataset {dataset_name}")
        return version_id
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get version information by ID."""
        if version_id not in self.metadata["versions"]:
            return None
        
        version_data = self.metadata["versions"][version_id]
        return DataVersion(
            version_id=version_id,
            dataset_name=version_data["dataset_name"],
            version_number=version_data["version_number"],
            creation_date=datetime.fromisoformat(version_data["creation_date"]),
            description=version_data["description"],
            changes=version_data["changes"],
            metrics=version_data["metrics"],
            file_path=version_data["file_path"],
            size_bytes=version_data["size_bytes"],
            checksum=version_data["checksum"]
        )
    
    def list_versions(self, dataset_name: str) -> List[DataVersion]:
        """List all versions of a dataset."""
        if dataset_name not in self.metadata["datasets"]:
            return []
        
        versions = []
        for version_id in self.metadata["datasets"][dataset_name]["versions"]:
            version = self.get_version(version_id)
            if version:
                versions.append(version)
        
        return sorted(versions, key=lambda v: v.creation_date, reverse=True)
    
    def _calculate_size(self, path: Path) -> int:
        """Calculate total size of dataset."""
        if path.is_file():
            return path.stat().st_size
        else:
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate checksum for dataset."""
        if path.is_file():
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        else:
            files_info = []
            for file_path in sorted(path.rglob('*')):
                if file_path.is_file():
                    files_info.append(f"{file_path.name}:{file_path.stat().st_size}")
            
            combined = "|".join(files_info)
            return hashlib.md5(combined.encode()).hexdigest()


class DataHealthMonitor:
    """
    Monitors data health and generates alerts for data quality issues.
    """
    
    def __init__(self, 
                 quality_threshold: float = 0.7,
                 drift_threshold: float = 0.1,
                 freshness_threshold_hours: float = 24.0):
        """
        Initialize the data health monitor.
        
        Args:
            quality_threshold: Minimum acceptable quality score
            drift_threshold: Maximum acceptable drift score
            freshness_threshold_hours: Maximum acceptable data age in hours
        """
        self.quality_threshold = quality_threshold
        self.drift_threshold = drift_threshold
        self.freshness_threshold_hours = freshness_threshold_hours
        self.assessor = DataQualityAssessor()
        self.comparator = DatasetComparator()
    
    def monitor_dataset_health(self, 
                              dataset_path: Union[str, Path],
                              dataset_name: str,
                              baseline_path: Optional[Union[str, Path]] = None) -> DataHealthMetrics:
        """
        Monitor the health of a dataset and generate alerts.
        
        Args:
            dataset_path: Path to the dataset to monitor
            dataset_name: Name of the dataset
            baseline_path: Optional baseline dataset for drift detection
            
        Returns:
            DataHealthMetrics: Health metrics and alerts
        """
        dataset_path = Path(dataset_path)
        alerts = []
        
        # Assess data quality
        try:
            quality_report = self.assessor.assess_quality(dataset_path)
            quality_score = quality_report.overall_score
            
            if quality_score < self.quality_threshold:
                alerts.append(f"Quality score {quality_score:.2f} below threshold {self.quality_threshold}")
        
        except Exception as e:
            logger.error(f"Error assessing quality: {e}")
            quality_score = 0.0
            alerts.append(f"Quality assessment failed: {str(e)}")
        
        # Check for data drift if baseline is provided
        drift_score = 0.0
        if baseline_path:
            try:
                drift_metrics = self.comparator.detect_data_drift(baseline_path, dataset_path)
                drift_score = drift_metrics.drift_score
                
                if drift_metrics.drift_detected:
                    alerts.append(f"Data drift detected (score: {drift_score:.3f})")
                    alerts.extend([f"Affected feature: {feature}" for feature in drift_metrics.affected_features])
            
            except Exception as e:
                logger.error(f"Error detecting drift: {e}")
                alerts.append(f"Drift detection failed: {str(e)}")
        
        # Check data freshness
        try:
            latest_file = max(dataset_path.rglob('*'), key=lambda p: p.stat().st_mtime if p.is_file() else 0)
            if latest_file.is_file():
                age_hours = (datetime.now().timestamp() - latest_file.stat().st_mtime) / 3600
                if age_hours > self.freshness_threshold_hours:
                    alerts.append(f"Data is {age_hours:.1f} hours old (threshold: {self.freshness_threshold_hours})")
            else:
                age_hours = float('inf')
                alerts.append("No files found in dataset")
        except Exception as e:
            logger.error(f"Error checking freshness: {e}")
            age_hours = float('inf')
            alerts.append(f"Freshness check failed: {str(e)}")
        
        # Calculate additional metrics
        try:
            total_samples = len(list(dataset_path.rglob('*.jpg'))) + len(list(dataset_path.rglob('*.png')))
            anomaly_count = len([issue for issue in quality_report.quality_issues if issue.severity in ['high', 'critical']])
            missing_data_percentage = 0.0  # Placeholder - would need specific implementation
            duplicate_percentage = (quality_report.dataset_statistics.duplicate_count / total_samples * 100) if total_samples > 0 else 0.0
            format_consistency_score = 1.0 - (len(quality_report.dataset_statistics.format_distribution) - 1) / max(len(quality_report.dataset_statistics.format_distribution), 1)
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            total_samples = 0
            anomaly_count = 0
            missing_data_percentage = 0.0
            duplicate_percentage = 0.0
            format_consistency_score = 0.0
        
        return DataHealthMetrics(
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            total_samples=total_samples,
            quality_score=quality_score,
            drift_score=drift_score,
            anomaly_count=anomaly_count,
            missing_data_percentage=missing_data_percentage,
            duplicate_percentage=duplicate_percentage,
            format_consistency_score=format_consistency_score,
            freshness_hours=age_hours,
            alerts=alerts
        )


class DataInsightsReporter:
    """
    Generates comprehensive data insights reports with visualizations.
    """
    
    def __init__(self, 
                 output_dir: str = "data_reports",
                 lineage_tracker: Optional[DataLineageTracker] = None,
                 version_manager: Optional[DataVersionManager] = None,
                 health_monitor: Optional[DataHealthMonitor] = None):
        """
        Initialize the insights reporter.
        
        Args:
            output_dir: Directory to save reports and visualizations
            lineage_tracker: Optional lineage tracker instance
            version_manager: Optional version manager instance
            health_monitor: Optional health monitor instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.lineage_tracker = lineage_tracker or DataLineageTracker()
        self.version_manager = version_manager or DataVersionManager()
        self.health_monitor = health_monitor or DataHealthMonitor()
        
        self.assessor = DataQualityAssessor()
        self.comparator = DatasetComparator()
    
    def generate_comprehensive_report(self, 
                                    dataset_path: Union[str, Path],
                                    dataset_name: str,
                                    baseline_path: Optional[Union[str, Path]] = None,
                                    synthetic_path: Optional[Union[str, Path]] = None) -> DataInsightReport:
        """
        Generate a comprehensive data insights report.
        
        Args:
            dataset_path: Path to the dataset
            dataset_name: Name of the dataset
            baseline_path: Optional baseline dataset for drift analysis
            synthetic_path: Optional synthetic dataset for comparison
            
        Returns:
            DataInsightReport: Comprehensive insights report
        """
        dataset_path = Path(dataset_path)
        report_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Generating comprehensive report for {dataset_name}")
        
        # Dataset summary
        dataset_summary = self._generate_dataset_summary(dataset_path, dataset_name)
        
        # Quality analysis
        quality_analysis = self.assessor.assess_quality(dataset_path)
        
        # Distribution analysis (if synthetic data provided)
        distribution_analysis = None
        if synthetic_path:
            try:
                distribution_analysis = self.comparator.compare_datasets(dataset_path, synthetic_path)
            except Exception as e:
                logger.warning(f"Could not perform distribution analysis: {e}")
        
        # Drift analysis (if baseline provided)
        drift_analysis = None
        if baseline_path:
            try:
                drift_analysis = self.comparator.detect_data_drift(baseline_path, dataset_path)
            except Exception as e:
                logger.warning(f"Could not perform drift analysis: {e}")
        
        # Lineage information
        dataset_id = self.lineage_tracker.register_dataset(dataset_path, metadata={"name": dataset_name})
        lineage_info = self.lineage_tracker.get_lineage(dataset_id)
        
        # Health metrics
        health_metrics = self.health_monitor.monitor_dataset_health(dataset_path, dataset_name, baseline_path)
        
        # Generate recommendations
        recommendations = self._generate_comprehensive_recommendations(
            quality_analysis, distribution_analysis, drift_analysis, health_metrics
        )
        
        # Create visualizations
        visualizations = self._create_report_visualizations(
            report_id, quality_analysis, distribution_analysis, drift_analysis, health_metrics
        )
        
        # Create report
        report = DataInsightReport(
            report_id=report_id,
            generation_timestamp=datetime.now(),
            dataset_summary=dataset_summary,
            quality_analysis=quality_analysis,
            distribution_analysis=distribution_analysis,
            drift_analysis=drift_analysis,
            lineage_info=lineage_info,
            health_metrics=health_metrics,
            recommendations=recommendations,
            visualizations=visualizations
        )
        
        # Save report
        self._save_report(report)
        
        logger.info(f"Generated comprehensive report: {report_id}")
        return report
    
    def _generate_dataset_summary(self, dataset_path: Path, dataset_name: str) -> Dict[str, Any]:
        """Generate a summary of the dataset."""
        try:
            # Count files by type
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(dataset_path.glob(f"**/*{ext}"))
                image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
            
            # Get date range
            file_dates = []
            for f in dataset_path.rglob('*'):
                if f.is_file():
                    file_dates.append(datetime.fromtimestamp(f.stat().st_mtime))
            
            date_range = (min(file_dates), max(file_dates)) if file_dates else (None, None)
            
            return {
                "name": dataset_name,
                "path": str(dataset_path),
                "total_files": len(list(dataset_path.rglob('*'))),
                "image_files": len(image_files),
                "total_size_mb": total_size / (1024 * 1024),
                "date_range": {
                    "start": date_range[0].isoformat() if date_range[0] else None,
                    "end": date_range[1].isoformat() if date_range[1] else None
                },
                "directory_structure": self._analyze_directory_structure(dataset_path)
            }
        
        except Exception as e:
            logger.error(f"Error generating dataset summary: {e}")
            return {"name": dataset_name, "path": str(dataset_path), "error": str(e)}
    
    def _analyze_directory_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyze the directory structure of the dataset."""
        structure = {"directories": 0, "files_by_directory": {}}
        
        for item in dataset_path.rglob('*'):
            if item.is_dir():
                structure["directories"] += 1
            else:
                parent_dir = str(item.parent.relative_to(dataset_path))
                if parent_dir not in structure["files_by_directory"]:
                    structure["files_by_directory"][parent_dir] = 0
                structure["files_by_directory"][parent_dir] += 1
        
        return structure
    
    def _generate_comprehensive_recommendations(self, 
                                              quality_analysis: QualityReport,
                                              distribution_analysis: Optional[DistributionComparison],
                                              drift_analysis: Optional[DataDriftMetrics],
                                              health_metrics: DataHealthMetrics) -> List[str]:
        """Generate comprehensive recommendations based on all analyses."""
        recommendations = []
        
        # Quality-based recommendations
        recommendations.extend(quality_analysis.recommendations)
        
        # Distribution-based recommendations
        if distribution_analysis and distribution_analysis.distribution_match_score < 0.8:
            recommendations.append("Synthetic data distribution does not closely match real data - consider adjusting generation parameters")
            
            if distribution_analysis.visual_similarity_score < 0.7:
                recommendations.append("Visual similarity between synthetic and real data is low - review ControlNet conditioning")
        
        # Drift-based recommendations
        if drift_analysis:
            recommendations.extend(drift_analysis.recommendations)
        
        # Health-based recommendations
        if health_metrics.alerts:
            recommendations.append("Address data health alerts to maintain system reliability")
            
            if health_metrics.quality_score < 0.7:
                recommendations.append("Implement automated quality filtering to improve dataset quality")
            
            if health_metrics.freshness_hours > 48:
                recommendations.append("Data is becoming stale - consider implementing automated data refresh")
        
        # Performance recommendations
        if health_metrics.total_samples < 1000:
            recommendations.append("Dataset size is small - consider data augmentation or synthetic data generation")
        
        if health_metrics.duplicate_percentage > 10:
            recommendations.append("High duplicate percentage detected - implement deduplication pipeline")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _create_report_visualizations(self, 
                                    report_id: str,
                                    quality_analysis: QualityReport,
                                    distribution_analysis: Optional[DistributionComparison],
                                    drift_analysis: Optional[DataDriftMetrics],
                                    health_metrics: DataHealthMetrics) -> Dict[str, str]:
        """Create visualizations for the report."""
        visualizations = {}
        viz_dir = self.output_dir / f"{report_id}_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Quality visualization
        try:
            quality_viz_path = viz_dir / "quality_analysis.png"
            self._create_quality_visualization(quality_analysis, str(quality_viz_path))
            visualizations["quality_analysis"] = str(quality_viz_path)
        except Exception as e:
            logger.warning(f"Could not create quality visualization: {e}")
        
        # Distribution comparison visualization
        if distribution_analysis:
            try:
                dist_viz_path = viz_dir / "distribution_comparison.png"
                # This would need the actual feature data - placeholder for now
                visualizations["distribution_comparison"] = str(dist_viz_path)
            except Exception as e:
                logger.warning(f"Could not create distribution visualization: {e}")
        
        # Drift visualization
        if drift_analysis:
            try:
                drift_viz_path = viz_dir / "drift_analysis.png"
                self._create_drift_visualization(drift_analysis, str(drift_viz_path))
                visualizations["drift_analysis"] = str(drift_viz_path)
            except Exception as e:
                logger.warning(f"Could not create drift visualization: {e}")
        
        # Health dashboard
        try:
            health_viz_path = viz_dir / "health_dashboard.png"
            self._create_health_dashboard(health_metrics, str(health_viz_path))
            visualizations["health_dashboard"] = str(health_viz_path)
        except Exception as e:
            logger.warning(f"Could not create health dashboard: {e}")
        
        return visualizations
    
    def _create_quality_visualization(self, quality_analysis: QualityReport, output_path: str):
        """Create quality analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Data Quality Analysis (Score: {quality_analysis.overall_score:.2f})', fontsize=16)
        
        # Quality issues
        if quality_analysis.quality_issues:
            issue_types = [issue.issue_type for issue in quality_analysis.quality_issues]
            issue_counts = [issue.affected_count for issue in quality_analysis.quality_issues]
            axes[0, 0].bar(issue_types, issue_counts)
            axes[0, 0].set_title('Quality Issues')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Format distribution
        if quality_analysis.dataset_statistics.format_distribution:
            format_data = quality_analysis.dataset_statistics.format_distribution
            axes[0, 1].pie(format_data.values(), labels=format_data.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('Format Distribution')
        
        # Contrast and clarity
        axes[1, 0].bar(['Low Contrast', 'High Contrast'], 
                       [quality_analysis.contrast_analysis.low_contrast_count,
                        quality_analysis.contrast_analysis.high_contrast_count])
        axes[1, 0].set_title('Contrast Analysis')
        
        axes[1, 1].bar(['Blurry', 'Sharp'], 
                       [quality_analysis.clarity_analysis.blurry_images_count,
                        quality_analysis.clarity_analysis.sharp_images_count])
        axes[1, 1].set_title('Clarity Analysis')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_drift_visualization(self, drift_analysis: DataDriftMetrics, output_path: str):
        """Create drift analysis visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Drift scores by feature
        features = list(drift_analysis.drift_magnitude.keys())
        scores = list(drift_analysis.drift_magnitude.values())
        
        colors = ['red' if score > drift_analysis.drift_threshold else 'green' for score in scores]
        
        ax1.bar(features, scores, color=colors, alpha=0.7)
        ax1.axhline(y=drift_analysis.drift_threshold, color='red', linestyle='--', 
                    label=f'Drift Threshold ({drift_analysis.drift_threshold})')
        ax1.set_title('Data Drift by Feature')
        ax1.set_xlabel('Feature')
        ax1.set_ylabel('Drift Score')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Overall drift status
        status_text = "DRIFT DETECTED" if drift_analysis.drift_detected else "NO DRIFT"
        status_color = "red" if drift_analysis.drift_detected else "green"
        
        ax2.text(0.5, 0.7, status_text, ha='center', va='center', 
                 fontsize=20, fontweight='bold', color=status_color,
                 transform=ax2.transAxes)
        
        ax2.text(0.5, 0.5, f"Overall Score: {drift_analysis.drift_score:.3f}", 
                 ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        
        ax2.set_title('Drift Detection Summary')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_health_dashboard(self, health_metrics: DataHealthMetrics, output_path: str):
        """Create health monitoring dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Data Health Dashboard - {health_metrics.dataset_name}', fontsize=16)
        
        # Quality score gauge
        self._create_gauge(axes[0, 0], health_metrics.quality_score, "Quality Score", 0, 1)
        
        # Drift score gauge
        self._create_gauge(axes[0, 1], health_metrics.drift_score, "Drift Score", 0, 0.5)
        
        # Freshness indicator
        freshness_score = max(0, 1 - health_metrics.freshness_hours / 168)  # 1 week = 168 hours
        self._create_gauge(axes[0, 2], freshness_score, "Freshness", 0, 1)
        
        # Sample count
        axes[1, 0].bar(['Total Samples'], [health_metrics.total_samples])
        axes[1, 0].set_title('Dataset Size')
        axes[1, 0].set_ylabel('Count')
        
        # Issues summary
        issue_data = [
            health_metrics.anomaly_count,
            health_metrics.duplicate_percentage,
            health_metrics.missing_data_percentage
        ]
        issue_labels = ['Anomalies', 'Duplicates %', 'Missing %']
        axes[1, 1].bar(issue_labels, issue_data)
        axes[1, 1].set_title('Data Issues')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Alerts
        if health_metrics.alerts:
            alert_text = '\n'.join(health_metrics.alerts[:5])  # Show first 5 alerts
            axes[1, 2].text(0.1, 0.9, alert_text, transform=axes[1, 2].transAxes,
                            fontsize=10, verticalalignment='top', wrap=True)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Alerts', ha='center', va='center',
                            fontsize=14, color='green', transform=axes[1, 2].transAxes)
        
        axes[1, 2].set_title('Active Alerts')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_gauge(self, ax, value: float, title: str, min_val: float, max_val: float):
        """Create a gauge visualization."""
        # Normalize value to 0-1 range
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax.plot(theta, r, 'lightgray', linewidth=10)
        
        # Value arc
        value_theta = theta[:int(normalized * len(theta))]
        value_r = r[:int(normalized * len(theta))]
        
        # Color based on value
        if normalized < 0.3:
            color = 'red'
        elif normalized < 0.7:
            color = 'orange'
        else:
            color = 'green'
        
        ax.plot(value_theta, value_r, color, linewidth=10)
        
        # Add value text
        ax.text(0, -0.3, f'{value:.3f}', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0, -0.5, title, ha='center', va='center', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _save_report(self, report: DataInsightReport):
        """Save the report to file."""
        report_file = self.output_dir / f"{report.report_id}_report.json"
        
        # Convert report to dictionary for JSON serialization
        report_dict = {
            "report_id": report.report_id,
            "generation_timestamp": report.generation_timestamp.isoformat(),
            "dataset_summary": report.dataset_summary,
            "quality_analysis": {
                "overall_score": report.quality_analysis.overall_score,
                "total_images": report.quality_analysis.dataset_statistics.total_images,
                "quality_issues": [
                    {
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "description": issue.description,
                        "affected_count": issue.affected_count,
                        "percentage": issue.percentage
                    }
                    for issue in report.quality_analysis.quality_issues
                ]
            },
            "distribution_analysis": {
                "distribution_match_score": report.distribution_analysis.distribution_match_score,
                "visual_similarity_score": report.distribution_analysis.visual_similarity_score,
                "kl_divergence": report.distribution_analysis.kl_divergence,
                "wasserstein_distance": report.distribution_analysis.wasserstein_distance
            } if report.distribution_analysis else None,
            "drift_analysis": {
                "drift_detected": report.drift_analysis.drift_detected,
                "drift_score": report.drift_analysis.drift_score,
                "affected_features": report.drift_analysis.affected_features
            } if report.drift_analysis else None,
            "health_metrics": {
                "quality_score": report.health_metrics.quality_score,
                "drift_score": report.health_metrics.drift_score,
                "total_samples": report.health_metrics.total_samples,
                "alerts": report.health_metrics.alerts
            },
            "recommendations": report.recommendations,
            "visualizations": report.visualizations
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Saved report to {report_file}")


# Convenience function for generating reports
def generate_data_insights_report(dataset_path: Union[str, Path],
                                 dataset_name: str,
                                 output_dir: str = "data_reports",
                                 baseline_path: Optional[Union[str, Path]] = None,
                                 synthetic_path: Optional[Union[str, Path]] = None) -> DataInsightReport:
    """
    Convenience function to generate a comprehensive data insights report.
    
    Args:
        dataset_path: Path to the dataset
        dataset_name: Name of the dataset
        output_dir: Directory to save reports
        baseline_path: Optional baseline dataset for drift analysis
        synthetic_path: Optional synthetic dataset for comparison
        
    Returns:
        DataInsightReport: Comprehensive insights report
    """
    reporter = DataInsightsReporter(output_dir=output_dir)
    return reporter.generate_comprehensive_report(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        baseline_path=baseline_path,
        synthetic_path=synthetic_path
    )