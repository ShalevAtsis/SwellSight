"""
Property-based tests for data quality assessment framework.

Tests Property 26: Data Quality Assessment
Validates Requirements 6.4, 6.5
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, assume
from typing import List, Tuple
import logging

from src.swellsight.evaluation.data_quality import (
    DataQualityAssessor, QualityReport, QualityIssue
)

logger = logging.getLogger(__name__)


class TestDataQualityAssessmentProperties:
    """Property-based tests for data quality assessment framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = DataQualityAssessor()
        self.temp_dir = None
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self, 
                           num_images: int,
                           image_qualities: List[str],
                           resolutions: List[Tuple[int, int]],
                           formats: List[str]) -> Path:
        """Create a test dataset with specified characteristics."""
        self.temp_dir = tempfile.mkdtemp()
        dataset_path = Path(self.temp_dir) / "test_dataset"
        dataset_path.mkdir(exist_ok=True)
        
        for i in range(num_images):
            # Select characteristics for this image
            quality = image_qualities[i % len(image_qualities)]
            resolution = resolutions[i % len(resolutions)]
            format_ext = formats[i % len(formats)]
            
            # Create image based on quality
            img = self.create_image_with_quality(resolution, quality)
            
            # Save image
            filename = f"image_{i:04d}.{format_ext}"
            filepath = dataset_path / filename
            
            if format_ext.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                cv2.imwrite(str(filepath), img)
        
        return dataset_path
    
    def create_image_with_quality(self, resolution: Tuple[int, int], quality: str) -> np.ndarray:
        """Create an image with specified quality characteristics."""
        width, height = resolution
        
        if quality == "high":
            # High quality: good contrast, sharp, ocean-like colors
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Add ocean-like blue tones
            img[:, :, 0] = np.clip(img[:, :, 0] * 0.3 + 100, 0, 255)  # Blue channel
            img[:, :, 1] = np.clip(img[:, :, 1] * 0.5 + 80, 0, 255)   # Green channel
            img[:, :, 2] = np.clip(img[:, :, 2] * 0.2 + 50, 0, 255)   # Red channel
            
        elif quality == "low_contrast":
            # Low contrast: narrow intensity range
            base_intensity = 128
            img = np.full((height, width, 3), base_intensity, dtype=np.uint8)
            noise = np.random.randint(-20, 20, (height, width, 3))
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
        elif quality == "blurry":
            # Blurry: apply Gaussian blur
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (15, 15), 5)
            
        elif quality == "no_ocean":
            # No ocean content: land/sky colors
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Make it more brown/green (land colors)
            img[:, :, 0] = np.clip(img[:, :, 0] * 0.2, 0, 255)      # Less blue
            img[:, :, 1] = np.clip(img[:, :, 1] * 0.8 + 50, 0, 255) # More green
            img[:, :, 2] = np.clip(img[:, :, 2] * 0.6 + 30, 0, 255) # Some red
            
        else:  # "corrupted" or default
            # Create a mostly black image (simulates corruption)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:10, :10] = 255  # Small white corner
        
        return img
    
    @given(
        num_images=st.integers(min_value=5, max_value=50),
        quality_mix=st.lists(
            st.sampled_from(["high", "low_contrast", "blurry", "no_ocean"]),
            min_size=1, max_size=4
        )
    )
    @settings(max_examples=10, deadline=30000)
    def test_quality_assessment_consistency(self, num_images: int, quality_mix: List[str]):
        """
        Property: Quality assessment should be consistent and deterministic.
        
        For any dataset, running quality assessment multiple times should
        produce the same results.
        """
        assume(len(quality_mix) > 0)
        
        # Create test dataset
        resolutions = [(640, 480), (1920, 1080)]
        formats = ["jpg", "png"]
        
        dataset_path = self.create_test_dataset(
            num_images=num_images,
            image_qualities=quality_mix,
            resolutions=resolutions,
            formats=formats
        )
        
        try:
            # Run assessment twice
            report1 = self.assessor.assess_quality(dataset_path)
            report2 = self.assessor.assess_quality(dataset_path)
            
            # Results should be identical
            assert report1.overall_score == report2.overall_score
            assert len(report1.quality_issues) == len(report2.quality_issues)
            assert report1.dataset_statistics.total_images == report2.dataset_statistics.total_images
            
            # Specific metrics should match
            assert report1.resolution_analysis.mean_width == report2.resolution_analysis.mean_width
            assert report1.resolution_analysis.mean_height == report2.resolution_analysis.mean_height
            
            logger.info(f"✓ Quality assessment consistency verified for {num_images} images")
            
        except Exception as e:
            pytest.fail(f"Quality assessment consistency test failed: {e}")
    
    @given(
        num_images=st.integers(min_value=10, max_value=30),
        low_quality_ratio=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=8, deadline=30000)
    def test_quality_score_correlation(self, num_images: int, low_quality_ratio: float):
        """
        Property: Quality score should correlate with actual quality issues.
        
        Datasets with more quality issues should have lower quality scores.
        """
        # Create dataset with controlled quality distribution
        num_low_quality = int(num_images * low_quality_ratio)
        num_high_quality = num_images - num_low_quality
        
        quality_mix = ["low_contrast", "blurry"] * num_low_quality + ["high"] * num_high_quality
        if not quality_mix:
            quality_mix = ["high"]
        
        resolutions = [(1280, 720)]
        formats = ["jpg"]
        
        dataset_path = self.create_test_dataset(
            num_images=num_images,
            image_qualities=quality_mix,
            resolutions=resolutions,
            formats=formats
        )
        
        try:
            report = self.assessor.assess_quality(dataset_path)
            
            # Quality score should be inversely related to quality issues
            total_affected = sum(issue.affected_count for issue in report.quality_issues)
            issue_ratio = total_affected / num_images if num_images > 0 else 0
            
            # Higher issue ratio should correlate with lower quality score
            if issue_ratio > 0.5:
                assert report.overall_score < 0.7, f"High issue ratio ({issue_ratio:.2f}) should result in low quality score, got {report.overall_score:.2f}"
            elif issue_ratio < 0.1:
                assert report.overall_score > 0.5, f"Low issue ratio ({issue_ratio:.2f}) should result in higher quality score, got {report.overall_score:.2f}"
            
            logger.info(f"✓ Quality score correlation verified: {issue_ratio:.2f} issue ratio → {report.overall_score:.2f} quality score")
            
        except Exception as e:
            pytest.fail(f"Quality score correlation test failed: {e}")
    
    @given(
        resolution_variety=st.lists(
            st.tuples(
                st.integers(min_value=320, max_value=3840),
                st.integers(min_value=240, max_value=2160)
            ),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=8, deadline=30000)
    def test_resolution_analysis_accuracy(self, resolution_variety: List[Tuple[int, int]]):
        """
        Property: Resolution analysis should accurately detect resolution issues.
        
        Images below 480p should be flagged as low resolution.
        """
        assume(len(resolution_variety) > 0)
        
        num_images = len(resolution_variety) * 2
        quality_mix = ["high"]
        formats = ["png"]
        
        dataset_path = self.create_test_dataset(
            num_images=num_images,
            image_qualities=quality_mix,
            resolutions=resolution_variety,
            formats=formats
        )
        
        try:
            report = self.assessor.assess_quality(dataset_path)
            
            # Count expected low resolution images
            expected_low_res = sum(1 for w, h in resolution_variety if w * h < 640 * 480) * 2
            
            # Check if low resolution issue is detected when expected
            low_res_issues = [issue for issue in report.quality_issues if issue.issue_type == "low_resolution"]
            
            if expected_low_res > 0:
                assert len(low_res_issues) > 0, "Low resolution images should be detected"
                assert report.resolution_analysis.below_480p_count >= expected_low_res, f"Expected {expected_low_res} low-res images, detected {report.resolution_analysis.below_480p_count}"
            
            # Resolution statistics should be reasonable
            assert report.resolution_analysis.mean_width > 0
            assert report.resolution_analysis.mean_height > 0
            assert report.resolution_analysis.std_width >= 0
            assert report.resolution_analysis.std_height >= 0
            
            logger.info(f"✓ Resolution analysis accuracy verified: {expected_low_res} expected low-res, {report.resolution_analysis.below_480p_count} detected")
            
        except Exception as e:
            pytest.fail(f"Resolution analysis accuracy test failed: {e}")
    
    @given(
        num_images=st.integers(min_value=5, max_value=20),
        ocean_content_ratio=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=8, deadline=30000)
    def test_ocean_coverage_detection(self, num_images: int, ocean_content_ratio: float):
        """
        Property: Ocean coverage detection should identify images without ocean content.
        
        Images without ocean content should be flagged appropriately.
        """
        # Create dataset with controlled ocean content
        num_no_ocean = int(num_images * (1 - ocean_content_ratio))
        num_with_ocean = num_images - num_no_ocean
        
        quality_mix = ["no_ocean"] * num_no_ocean + ["high"] * num_with_ocean
        if not quality_mix:
            quality_mix = ["high"]
        
        resolutions = [(1280, 720)]
        formats = ["jpg"]
        
        dataset_path = self.create_test_dataset(
            num_images=num_images,
            image_qualities=quality_mix,
            resolutions=resolutions,
            formats=formats
        )
        
        try:
            report = self.assessor.assess_quality(dataset_path)
            
            # Check ocean coverage analysis
            if num_no_ocean > 0:
                # Should detect some images with no ocean content
                no_ocean_issues = [issue for issue in report.quality_issues if issue.issue_type == "no_ocean_content"]
                if num_no_ocean > num_images * 0.1:  # Only check if significant portion
                    assert len(no_ocean_issues) > 0, "Images without ocean content should be detected"
            
            # Ocean coverage statistics should be reasonable
            assert 0 <= report.ocean_coverage_analysis.mean_ocean_coverage <= 1
            assert report.ocean_coverage_analysis.std_ocean_coverage >= 0
            assert report.ocean_coverage_analysis.min_ocean_coverage >= 0
            assert report.ocean_coverage_analysis.max_ocean_coverage <= 1
            
            logger.info(f"✓ Ocean coverage detection verified: {num_no_ocean} no-ocean images, mean coverage {report.ocean_coverage_analysis.mean_ocean_coverage:.2f}")
            
        except Exception as e:
            pytest.fail(f"Ocean coverage detection test failed: {e}")
    
    @given(
        format_variety=st.lists(
            st.sampled_from(["jpg", "png", "webp"]),
            min_size=1, max_size=3
        )
    )
    @settings(max_examples=6, deadline=30000)
    def test_format_support(self, format_variety: List[str]):
        """
        Property: Quality assessment should support multiple image formats.
        
        The system should handle JPEG, PNG, and WebP formats correctly.
        """
        assume(len(format_variety) > 0)
        
        num_images = len(format_variety) * 3
        quality_mix = ["high"]
        resolutions = [(1280, 720)]
        
        dataset_path = self.create_test_dataset(
            num_images=num_images,
            image_qualities=quality_mix,
            resolutions=resolutions,
            formats=format_variety
        )
        
        try:
            report = self.assessor.assess_quality(dataset_path)
            
            # Should successfully process all images
            assert report.dataset_statistics.total_images == num_images
            
            # Format distribution should reflect input
            format_dist = report.dataset_statistics.format_distribution
            for fmt in format_variety:
                expected_ext = f".{fmt}"
                assert expected_ext in format_dist, f"Format {expected_ext} should be detected"
                assert format_dist[expected_ext] > 0, f"Should have images with format {expected_ext}"
            
            # No corruption issues should be detected for valid formats
            corruption_issues = [issue for issue in report.quality_issues if issue.issue_type == "corruption"]
            assert len(corruption_issues) == 0, "Valid format images should not be flagged as corrupted"
            
            logger.info(f"✓ Format support verified for formats: {format_variety}")
            
        except Exception as e:
            pytest.fail(f"Format support test failed: {e}")
    
    def test_empty_dataset_handling(self):
        """
        Property: Quality assessment should handle empty datasets gracefully.
        
        Empty datasets should be handled without crashing.
        """
        self.temp_dir = tempfile.mkdtemp()
        empty_dataset_path = Path(self.temp_dir) / "empty_dataset"
        empty_dataset_path.mkdir(exist_ok=True)
        
        try:
            # Should raise ValueError for empty dataset
            with pytest.raises(ValueError, match="No image files found"):
                self.assessor.assess_quality(empty_dataset_path)
            
            logger.info("✓ Empty dataset handling verified")
            
        except Exception as e:
            if "No image files found" not in str(e):
                pytest.fail(f"Empty dataset handling test failed: {e}")
    
    @given(
        num_images=st.integers(min_value=3, max_value=15)
    )
    @settings(max_examples=5, deadline=30000)
    def test_quality_issue_severity_classification(self, num_images: int):
        """
        Property: Quality issues should be classified with appropriate severity levels.
        
        Critical issues (corruption, no ocean) should be marked as critical.
        Quality issues (blur, low contrast) should be marked appropriately.
        """
        # Create dataset with various quality issues
        quality_mix = ["high", "low_contrast", "blurry", "no_ocean"]
        resolutions = [(1280, 720)]
        formats = ["jpg"]
        
        dataset_path = self.create_test_dataset(
            num_images=num_images,
            image_qualities=quality_mix,
            resolutions=resolutions,
            formats=formats
        )
        
        try:
            report = self.assessor.assess_quality(dataset_path)
            
            # Check severity classification
            for issue in report.quality_issues:
                if issue.issue_type == "no_ocean_content":
                    assert issue.severity == "critical", "No ocean content should be critical severity"
                elif issue.issue_type == "corruption":
                    assert issue.severity == "critical", "Corruption should be critical severity"
                elif issue.issue_type == "blur":
                    assert issue.severity in ["high", "medium"], "Blur should be high or medium severity"
                elif issue.issue_type == "low_contrast":
                    assert issue.severity in ["medium", "low"], "Low contrast should be medium or low severity"
                
                # All issues should have valid severity levels
                assert issue.severity in ["low", "medium", "high", "critical"], f"Invalid severity: {issue.severity}"
                
                # Percentage should be reasonable
                assert 0 <= issue.percentage <= 100, f"Invalid percentage: {issue.percentage}"
                assert issue.affected_count >= 0, f"Invalid affected count: {issue.affected_count}"
            
            logger.info(f"✓ Quality issue severity classification verified for {len(report.quality_issues)} issues")
            
        except Exception as e:
            pytest.fail(f"Quality issue severity classification test failed: {e}")


if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v", "--tb=short"])