"""
Data Quality Assessment Framework for SwellSight Wave Analysis System.

This module provides comprehensive tools for assessing the quality of beach cam image datasets,
including statistical analysis, distribution analysis, and quality metrics computation.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mutual_info_score
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ResolutionStats:
    """Statistics about image resolutions in the dataset."""
    mean_width: float
    mean_height: float
    std_width: float
    std_height: float
    min_resolution: Tuple[int, int]
    max_resolution: Tuple[int, int]
    resolution_distribution: Dict[str, int]
    below_480p_count: int
    above_4k_count: int


@dataclass
class ContrastStats:
    """Statistics about image contrast and dynamic range."""
    mean_contrast: float
    std_contrast: float
    mean_dynamic_range: float
    std_dynamic_range: float
    low_contrast_count: int
    high_contrast_count: int
    histogram_entropy: float


@dataclass
class ClarityStats:
    """Statistics about image clarity and sharpness."""
    mean_sharpness: float
    std_sharpness: float
    mean_blur_score: float
    std_blur_score: float
    blurry_images_count: int
    sharp_images_count: int


@dataclass
class CoverageStats:
    """Statistics about ocean coverage in images."""
    mean_ocean_coverage: float
    std_ocean_coverage: float
    min_ocean_coverage: float
    max_ocean_coverage: float
    low_coverage_count: int
    no_ocean_count: int


@dataclass
class QualityIssue:
    """Represents a quality issue found in the dataset."""
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_count: int
    percentage: float
    recommendations: List[str]


@dataclass
class DatasetBalance:
    """Statistics about dataset balance across wave conditions."""
    wave_height_distribution: Dict[str, int]
    direction_distribution: Dict[str, int]
    breaking_type_distribution: Dict[str, int]
    weather_condition_distribution: Dict[str, int]
    time_of_day_distribution: Dict[str, int]
    balance_score: float  # 0-1, where 1 is perfectly balanced


@dataclass
class DatasetStatistics:
    """Overall dataset statistics."""
    total_images: int
    total_size_mb: float
    mean_file_size_kb: float
    format_distribution: Dict[str, int]
    creation_date_range: Tuple[datetime, datetime]
    duplicate_count: int
    corrupted_count: int


@dataclass
class QualityReport:
    """Comprehensive data quality assessment report."""
    overall_score: float
    resolution_analysis: ResolutionStats
    contrast_analysis: ContrastStats
    clarity_analysis: ClarityStats
    ocean_coverage_analysis: CoverageStats
    dataset_balance: DatasetBalance
    dataset_statistics: DatasetStatistics
    quality_issues: List[QualityIssue]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class DataQualityAssessor:
    """
    Comprehensive data quality assessment framework for beach cam image datasets.
    
    This class provides tools for analyzing image quality, dataset balance,
    and generating detailed quality reports with recommendations.
    """
    
    def __init__(self, 
                 min_resolution: Tuple[int, int] = (640, 480),
                 max_resolution: Tuple[int, int] = (3840, 2160),
                 min_contrast_threshold: float = 0.1,
                 min_sharpness_threshold: float = 100.0,
                 min_ocean_coverage: float = 0.3):
        """
        Initialize the data quality assessor.
        
        Args:
            min_resolution: Minimum acceptable resolution (width, height)
            max_resolution: Maximum acceptable resolution (width, height)
            min_contrast_threshold: Minimum contrast threshold
            min_sharpness_threshold: Minimum sharpness threshold
            min_ocean_coverage: Minimum ocean coverage ratio
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_contrast_threshold = min_contrast_threshold
        self.min_sharpness_threshold = min_sharpness_threshold
        self.min_ocean_coverage = min_ocean_coverage
        
    def assess_quality(self, dataset_path: Union[str, Path], 
                      labels_path: Optional[Union[str, Path]] = None) -> QualityReport:
        """
        Assess the quality of a dataset and generate a comprehensive report.
        
        Args:
            dataset_path: Path to the dataset directory
            labels_path: Optional path to labels file
            
        Returns:
            QualityReport: Comprehensive quality assessment report
        """
        dataset_path = Path(dataset_path)
        
        # Get all image files
        image_files = self._get_image_files(dataset_path)
        
        if not image_files:
            raise ValueError(f"No image files found in {dataset_path}")
        
        logger.info(f"Assessing quality of {len(image_files)} images...")
        
        # Analyze different aspects of data quality
        resolution_stats = self._analyze_resolutions(image_files)
        contrast_stats = self._analyze_contrast(image_files)
        clarity_stats = self._analyze_clarity(image_files)
        coverage_stats = self._analyze_ocean_coverage(image_files)
        dataset_stats = self._analyze_dataset_statistics(image_files)
        
        # Analyze dataset balance if labels are available
        if labels_path:
            dataset_balance = self._analyze_dataset_balance(labels_path)
        else:
            dataset_balance = self._create_empty_balance()
        
        # Identify quality issues
        quality_issues = self._identify_quality_issues(
            resolution_stats, contrast_stats, clarity_stats, 
            coverage_stats, dataset_stats
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_issues)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(
            resolution_stats, contrast_stats, clarity_stats, 
            coverage_stats, quality_issues
        )
        
        return QualityReport(
            overall_score=overall_score,
            resolution_analysis=resolution_stats,
            contrast_analysis=contrast_stats,
            clarity_analysis=clarity_stats,
            ocean_coverage_analysis=coverage_stats,
            dataset_balance=dataset_balance,
            dataset_statistics=dataset_stats,
            quality_issues=quality_issues,
            recommendations=recommendations
        )
    
    def _get_image_files(self, dataset_path: Path) -> List[Path]:
        """Get all image files from the dataset directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f"**/*{ext}"))
            image_files.extend(dataset_path.glob(f"**/*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _analyze_resolutions(self, image_files: List[Path]) -> ResolutionStats:
        """Analyze image resolutions in the dataset."""
        resolutions = []
        resolution_counts = {}
        below_480p = 0
        above_4k = 0
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                height, width = img.shape[:2]
                resolutions.append((width, height))
                
                # Count resolution categories
                res_key = f"{width}x{height}"
                resolution_counts[res_key] = resolution_counts.get(res_key, 0) + 1
                
                # Check for out-of-range resolutions
                if width * height < 640 * 480:  # Below 480p
                    below_480p += 1
                elif width * height > 3840 * 2160:  # Above 4K
                    above_4k += 1
                    
            except Exception as e:
                logger.warning(f"Error reading {img_path}: {e}")
                continue
        
        if not resolutions:
            raise ValueError("No valid images found for resolution analysis")
        
        widths, heights = zip(*resolutions)
        
        return ResolutionStats(
            mean_width=np.mean(widths),
            mean_height=np.mean(heights),
            std_width=np.std(widths),
            std_height=np.std(heights),
            min_resolution=min(resolutions, key=lambda x: x[0] * x[1]),
            max_resolution=max(resolutions, key=lambda x: x[0] * x[1]),
            resolution_distribution=resolution_counts,
            below_480p_count=below_480p,
            above_4k_count=above_4k
        )
    
    def _analyze_contrast(self, image_files: List[Path]) -> ContrastStats:
        """Analyze image contrast and dynamic range."""
        contrasts = []
        dynamic_ranges = []
        low_contrast = 0
        high_contrast = 0
        entropies = []
        
        for img_path in image_files[:100]:  # Sample for performance
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Calculate contrast (RMS contrast)
                contrast = np.std(img) / np.mean(img) if np.mean(img) > 0 else 0
                contrasts.append(contrast)
                
                # Calculate dynamic range
                dynamic_range = (np.max(img) - np.min(img)) / 255.0
                dynamic_ranges.append(dynamic_range)
                
                # Calculate histogram entropy
                hist, _ = np.histogram(img, bins=256, range=(0, 256))
                hist = hist / np.sum(hist)  # Normalize
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropies.append(entropy)
                
                # Count low/high contrast images
                if contrast < self.min_contrast_threshold:
                    low_contrast += 1
                elif contrast > 0.5:  # High contrast threshold
                    high_contrast += 1
                    
            except Exception as e:
                logger.warning(f"Error analyzing contrast for {img_path}: {e}")
                continue
        
        return ContrastStats(
            mean_contrast=np.mean(contrasts) if contrasts else 0,
            std_contrast=np.std(contrasts) if contrasts else 0,
            mean_dynamic_range=np.mean(dynamic_ranges) if dynamic_ranges else 0,
            std_dynamic_range=np.std(dynamic_ranges) if dynamic_ranges else 0,
            low_contrast_count=low_contrast,
            high_contrast_count=high_contrast,
            histogram_entropy=np.mean(entropies) if entropies else 0
        )
    
    def _analyze_clarity(self, image_files: List[Path]) -> ClarityStats:
        """Analyze image clarity and sharpness."""
        sharpness_scores = []
        blur_scores = []
        blurry_count = 0
        sharp_count = 0
        
        for img_path in image_files[:100]:  # Sample for performance
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Calculate sharpness using Laplacian variance
                laplacian = cv2.Laplacian(img, cv2.CV_64F)
                sharpness = laplacian.var()
                sharpness_scores.append(sharpness)
                
                # Calculate blur score using gradient magnitude
                grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                blur_score = np.mean(np.sqrt(grad_x**2 + grad_y**2))
                blur_scores.append(blur_score)
                
                # Count blurry/sharp images
                if sharpness < self.min_sharpness_threshold:
                    blurry_count += 1
                elif sharpness > 500:  # Sharp threshold
                    sharp_count += 1
                    
            except Exception as e:
                logger.warning(f"Error analyzing clarity for {img_path}: {e}")
                continue
        
        return ClarityStats(
            mean_sharpness=np.mean(sharpness_scores) if sharpness_scores else 0,
            std_sharpness=np.std(sharpness_scores) if sharpness_scores else 0,
            mean_blur_score=np.mean(blur_scores) if blur_scores else 0,
            std_blur_score=np.std(blur_scores) if blur_scores else 0,
            blurry_images_count=blurry_count,
            sharp_images_count=sharp_count
        )
    
    def _analyze_ocean_coverage(self, image_files: List[Path]) -> CoverageStats:
        """Analyze ocean coverage in images using simple color-based detection."""
        coverage_ratios = []
        low_coverage = 0
        no_ocean = 0
        
        for img_path in image_files[:50]:  # Sample for performance
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Convert to HSV for better water detection
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Define blue/cyan range for water (rough approximation)
                lower_water = np.array([90, 50, 50])
                upper_water = np.array([130, 255, 255])
                
                # Create mask for water regions
                water_mask = cv2.inRange(hsv, lower_water, upper_water)
                
                # Calculate coverage ratio
                total_pixels = img.shape[0] * img.shape[1]
                water_pixels = np.sum(water_mask > 0)
                coverage_ratio = water_pixels / total_pixels
                
                coverage_ratios.append(coverage_ratio)
                
                # Count low/no coverage images
                if coverage_ratio < self.min_ocean_coverage:
                    low_coverage += 1
                if coverage_ratio < 0.1:
                    no_ocean += 1
                    
            except Exception as e:
                logger.warning(f"Error analyzing ocean coverage for {img_path}: {e}")
                continue
        
        return CoverageStats(
            mean_ocean_coverage=np.mean(coverage_ratios) if coverage_ratios else 0,
            std_ocean_coverage=np.std(coverage_ratios) if coverage_ratios else 0,
            min_ocean_coverage=np.min(coverage_ratios) if coverage_ratios else 0,
            max_ocean_coverage=np.max(coverage_ratios) if coverage_ratios else 0,
            low_coverage_count=low_coverage,
            no_ocean_count=no_ocean
        )
    
    def _analyze_dataset_statistics(self, image_files: List[Path]) -> DatasetStatistics:
        """Analyze overall dataset statistics."""
        total_size = 0
        file_sizes = []
        format_counts = {}
        creation_dates = []
        duplicates = 0
        corrupted = 0
        
        for img_path in image_files:
            try:
                # File size
                size = img_path.stat().st_size
                total_size += size
                file_sizes.append(size / 1024)  # Convert to KB
                
                # Format
                ext = img_path.suffix.lower()
                format_counts[ext] = format_counts.get(ext, 0) + 1
                
                # Creation date
                creation_date = datetime.fromtimestamp(img_path.stat().st_ctime)
                creation_dates.append(creation_date)
                
                # Check if image is readable (corruption check)
                img = cv2.imread(str(img_path))
                if img is None:
                    corrupted += 1
                    
            except Exception as e:
                logger.warning(f"Error analyzing statistics for {img_path}: {e}")
                corrupted += 1
                continue
        
        # Simple duplicate detection based on file size
        size_counts = {}
        for size in file_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        duplicates = sum(count - 1 for count in size_counts.values() if count > 1)
        
        date_range = (min(creation_dates), max(creation_dates)) if creation_dates else (datetime.now(), datetime.now())
        
        return DatasetStatistics(
            total_images=len(image_files),
            total_size_mb=total_size / (1024 * 1024),
            mean_file_size_kb=np.mean(file_sizes) if file_sizes else 0,
            format_distribution=format_counts,
            creation_date_range=date_range,
            duplicate_count=duplicates,
            corrupted_count=corrupted
        )
    
    def _analyze_dataset_balance(self, labels_path: Path) -> DatasetBalance:
        """Analyze dataset balance across wave conditions."""
        # This is a placeholder - actual implementation would depend on label format
        return DatasetBalance(
            wave_height_distribution={},
            direction_distribution={},
            breaking_type_distribution={},
            weather_condition_distribution={},
            time_of_day_distribution={},
            balance_score=0.0
        )
    
    def _create_empty_balance(self) -> DatasetBalance:
        """Create empty dataset balance when labels are not available."""
        return DatasetBalance(
            wave_height_distribution={},
            direction_distribution={},
            breaking_type_distribution={},
            weather_condition_distribution={},
            time_of_day_distribution={},
            balance_score=0.0
        )
    
    def _identify_quality_issues(self, resolution_stats: ResolutionStats,
                               contrast_stats: ContrastStats,
                               clarity_stats: ClarityStats,
                               coverage_stats: CoverageStats,
                               dataset_stats: DatasetStatistics) -> List[QualityIssue]:
        """Identify quality issues in the dataset."""
        issues = []
        total_images = dataset_stats.total_images
        
        # Resolution issues
        if resolution_stats.below_480p_count > 0:
            issues.append(QualityIssue(
                issue_type="low_resolution",
                severity="high",
                description=f"{resolution_stats.below_480p_count} images below 480p resolution",
                affected_count=resolution_stats.below_480p_count,
                percentage=(resolution_stats.below_480p_count / total_images) * 100,
                recommendations=["Remove or upscale low-resolution images", "Check data collection pipeline"]
            ))
        
        # Contrast issues
        if contrast_stats.low_contrast_count > total_images * 0.1:
            issues.append(QualityIssue(
                issue_type="low_contrast",
                severity="medium",
                description=f"{contrast_stats.low_contrast_count} images with low contrast",
                affected_count=contrast_stats.low_contrast_count,
                percentage=(contrast_stats.low_contrast_count / total_images) * 100,
                recommendations=["Apply histogram equalization", "Check camera settings", "Filter out low-contrast images"]
            ))
        
        # Clarity issues
        if clarity_stats.blurry_images_count > total_images * 0.15:
            issues.append(QualityIssue(
                issue_type="blur",
                severity="high",
                description=f"{clarity_stats.blurry_images_count} blurry images detected",
                affected_count=clarity_stats.blurry_images_count,
                percentage=(clarity_stats.blurry_images_count / total_images) * 100,
                recommendations=["Remove blurry images", "Check camera focus", "Apply sharpening filters"]
            ))
        
        # Ocean coverage issues
        if coverage_stats.no_ocean_count > 0:
            issues.append(QualityIssue(
                issue_type="no_ocean_content",
                severity="critical",
                description=f"{coverage_stats.no_ocean_count} images with no detectable ocean content",
                affected_count=coverage_stats.no_ocean_count,
                percentage=(coverage_stats.no_ocean_count / total_images) * 100,
                recommendations=["Remove non-ocean images", "Improve ocean detection algorithm", "Review data collection criteria"]
            ))
        
        # Corruption issues
        if dataset_stats.corrupted_count > 0:
            issues.append(QualityIssue(
                issue_type="corruption",
                severity="critical",
                description=f"{dataset_stats.corrupted_count} corrupted or unreadable images",
                affected_count=dataset_stats.corrupted_count,
                percentage=(dataset_stats.corrupted_count / total_images) * 100,
                recommendations=["Remove corrupted images", "Check data transfer integrity", "Validate file formats"]
            ))
        
        return issues
    
    def _generate_recommendations(self, quality_issues: List[QualityIssue]) -> List[str]:
        """Generate overall recommendations based on quality issues."""
        recommendations = []
        
        # High-level recommendations based on issue patterns
        critical_issues = [issue for issue in quality_issues if issue.severity == "critical"]
        high_issues = [issue for issue in quality_issues if issue.severity == "high"]
        
        if critical_issues:
            recommendations.append("Address critical issues immediately before using dataset for training")
        
        if high_issues:
            recommendations.append("Resolve high-severity issues to improve model performance")
        
        # Specific recommendations
        issue_types = {issue.issue_type for issue in quality_issues}
        
        if "low_resolution" in issue_types or "blur" in issue_types:
            recommendations.append("Implement image quality filtering in data preprocessing pipeline")
        
        if "no_ocean_content" in issue_types:
            recommendations.append("Improve automated ocean region detection and filtering")
        
        if "corruption" in issue_types:
            recommendations.append("Add data integrity checks to data collection pipeline")
        
        if not quality_issues:
            recommendations.append("Dataset quality is good - proceed with training")
        
        return recommendations
    
    def _calculate_overall_score(self, resolution_stats: ResolutionStats,
                               contrast_stats: ContrastStats,
                               clarity_stats: ClarityStats,
                               coverage_stats: CoverageStats,
                               quality_issues: List[QualityIssue]) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0
        
        # Penalize based on quality issues
        for issue in quality_issues:
            if issue.severity == "critical":
                score -= 0.3 * (issue.percentage / 100)
            elif issue.severity == "high":
                score -= 0.2 * (issue.percentage / 100)
            elif issue.severity == "medium":
                score -= 0.1 * (issue.percentage / 100)
            else:  # low
                score -= 0.05 * (issue.percentage / 100)
        
        # Bonus for good metrics
        if contrast_stats.mean_contrast > 0.2:
            score += 0.05
        if clarity_stats.mean_sharpness > 200:
            score += 0.05
        if coverage_stats.mean_ocean_coverage > 0.5:
            score += 0.05
        
        return max(0.0, min(1.0, score))


def create_quality_visualization(quality_report: QualityReport, 
                               output_path: Optional[str] = None) -> None:
    """
    Create visualizations for the quality report.
    
    Args:
        quality_report: Quality report to visualize
        output_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Dataset Quality Report (Score: {quality_report.overall_score:.2f})', fontsize=16)
    
    # Resolution distribution
    if quality_report.resolution_analysis.resolution_distribution:
        res_data = quality_report.resolution_analysis.resolution_distribution
        axes[0, 0].bar(range(len(res_data)), list(res_data.values()))
        axes[0, 0].set_title('Resolution Distribution')
        axes[0, 0].set_xlabel('Resolution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Quality issues
    if quality_report.quality_issues:
        issue_types = [issue.issue_type for issue in quality_report.quality_issues]
        issue_counts = [issue.affected_count for issue in quality_report.quality_issues]
        axes[0, 1].bar(issue_types, issue_counts)
        axes[0, 1].set_title('Quality Issues')
        axes[0, 1].set_xlabel('Issue Type')
        axes[0, 1].set_ylabel('Affected Images')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Format distribution
    if quality_report.dataset_statistics.format_distribution:
        format_data = quality_report.dataset_statistics.format_distribution
        axes[0, 2].pie(format_data.values(), labels=format_data.keys(), autopct='%1.1f%%')
        axes[0, 2].set_title('Format Distribution')
    
    # Contrast analysis
    axes[1, 0].bar(['Low Contrast', 'High Contrast'], 
                   [quality_report.contrast_analysis.low_contrast_count,
                    quality_report.contrast_analysis.high_contrast_count])
    axes[1, 0].set_title('Contrast Analysis')
    axes[1, 0].set_ylabel('Count')
    
    # Clarity analysis
    axes[1, 1].bar(['Blurry', 'Sharp'], 
                   [quality_report.clarity_analysis.blurry_images_count,
                    quality_report.clarity_analysis.sharp_images_count])
    axes[1, 1].set_title('Clarity Analysis')
    axes[1, 1].set_ylabel('Count')
    
    # Ocean coverage
    coverage_data = [
        quality_report.ocean_coverage_analysis.no_ocean_count,
        quality_report.ocean_coverage_analysis.low_coverage_count,
        quality_report.dataset_statistics.total_images - 
        quality_report.ocean_coverage_analysis.no_ocean_count - 
        quality_report.ocean_coverage_analysis.low_coverage_count
    ]
    axes[1, 2].bar(['No Ocean', 'Low Coverage', 'Good Coverage'], coverage_data)
    axes[1, 2].set_title('Ocean Coverage Analysis')
    axes[1, 2].set_ylabel('Count')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()