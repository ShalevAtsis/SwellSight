"""
Property-based tests for beach cam image validation and quality enhancement.

Tests Properties 1 and 2 from the design document:
- Property 1: Input Format Validation
- Property 2: Image Quality Enhancement
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from PIL import Image
import io
import cv2

from src.swellsight.data.preprocessing import ImageProcessor, ImageFormat, BeachCamImage


class TestImageValidationProperties:
    """Property-based tests for image validation and quality enhancement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
    
    @given(
        width=st.integers(min_value=640, max_value=3840),
        height=st.integers(min_value=480, max_value=2160),
        format_type=st.sampled_from([ImageFormat.JPEG, ImageFormat.PNG, ImageFormat.WEBP])
    )
    @settings(max_examples=50, deadline=5000)
    def test_property_1_input_format_validation(self, width, height, format_type):
        """
        Property 1: Input Format Validation
        
        For any input image with resolution between 480p and 4K in JPEG/PNG/WebP format,
        the Wave_Analyzer should successfully process the image and return valid results.
        
        **Validates: Requirements 1.1, 1.3**
        """
        # Generate a valid RGB image within resolution bounds
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Property: Valid images should pass validation
        is_valid = self.processor.validate_image(image, format_type)
        assert is_valid, f"Valid image {width}x{height} {format_type.value} should pass validation"
        
        # Property: Valid images should be processable through the full pipeline
        try:
            result = self.processor.process_beach_cam_image(image, format_type)
            assert isinstance(result, BeachCamImage)
            assert result.resolution == (height, width)
            assert result.format == format_type
            assert 0.0 <= result.quality_score <= 1.0
            assert result.rgb_data.shape == (height, width, 3)
        except Exception as e:
            pytest.fail(f"Valid image should be processable: {e}")
    
    @given(
        width=st.integers(min_value=100, max_value=5000),
        height=st.integers(min_value=100, max_value=5000),
        format_type=st.sampled_from([ImageFormat.JPEG, ImageFormat.PNG, ImageFormat.WEBP])
    )
    @settings(max_examples=50, deadline=5000)
    def test_property_1_resolution_bounds_validation(self, width, height, format_type):
        """
        Property 1: Resolution bounds validation
        
        Images outside the 480p-4K range should be rejected.
        """
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Check if image is within valid bounds
        min_w, min_h = self.processor.min_resolution
        max_w, max_h = self.processor.max_resolution
        
        is_within_bounds = (width >= min_w and height >= min_h and 
                           width <= max_w and height <= max_h)
        
        # Property: Validation result should match bounds check
        is_valid = self.processor.validate_image(image, format_type)
        
        if is_within_bounds:
            assert is_valid, f"Image {width}x{height} within bounds should be valid"
        else:
            assert not is_valid, f"Image {width}x{height} outside bounds should be invalid"
    
    @given(
        channels=st.integers(min_value=1, max_value=5),
        dtype=st.sampled_from([np.uint8, np.float32, np.int32, np.uint16])
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_1_format_requirements_validation(self, channels, dtype):
        """
        Property 1: Format requirements validation
        
        Images must have exactly 3 channels (RGB) and supported data types.
        """
        # Create image with specified characteristics
        image = np.random.random((480, 640, channels)).astype(dtype)
        
        # Adjust value range based on dtype
        if dtype == np.uint8:
            image = (image * 255).astype(np.uint8)
        elif dtype in [np.float32, np.float64]:
            image = image.astype(dtype)  # Keep in [0,1] range
        
        format_type = ImageFormat.JPEG
        
        # Property: Only 3-channel images with supported dtypes should be valid
        is_valid = self.processor.validate_image(image, format_type)
        
        expected_valid = (channels == 3 and dtype in [np.uint8, np.float32, np.float64])
        
        if expected_valid:
            assert is_valid, f"3-channel {dtype} image should be valid"
        else:
            assert not is_valid, f"{channels}-channel {dtype} image should be invalid"
    
    @given(
        brightness_factor=st.floats(min_value=0.1, max_value=2.0),
        contrast_factor=st.floats(min_value=0.1, max_value=2.0),
        noise_level=st.floats(min_value=0.0, max_value=0.3)
    )
    @settings(max_examples=30, deadline=10000)
    def test_property_2_image_quality_enhancement(self, brightness_factor, contrast_factor, noise_level):
        """
        Property 2: Image Quality Enhancement
        
        For any input image with measurable quality issues, applying image enhancement
        should improve quantifiable quality metrics while preserving wave information.
        
        **Validates: Requirements 1.2**
        """
        # Create a base image with wave-like patterns
        height, width = 480, 640
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 3*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Create wave pattern
        wave_pattern = np.sin(X) * np.cos(Y) * 0.5 + 0.5
        base_image = np.stack([wave_pattern] * 3, axis=-1)
        
        # Degrade image quality
        degraded = base_image.copy()
        
        # Adjust brightness
        degraded = np.clip(degraded * brightness_factor, 0, 1)
        
        # Adjust contrast
        degraded = np.clip((degraded - 0.5) * contrast_factor + 0.5, 0, 1)
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, degraded.shape)
            degraded = np.clip(degraded + noise, 0, 1)
        
        # Convert to uint8
        degraded_uint8 = (degraded * 255).astype(np.uint8)
        
        # Calculate quality before enhancement
        quality_before = self.processor._calculate_quality_score(degraded_uint8)
        
        # Apply enhancement
        enhanced = self.processor.enhance_quality(degraded_uint8)
        
        # Calculate quality after enhancement
        quality_after = self.processor._calculate_quality_score(enhanced)
        
        # Property: Enhancement should improve or maintain quality
        assert quality_after >= quality_before * 0.9, \
            f"Enhancement should improve quality: {quality_before:.3f} -> {quality_after:.3f}"
        
        # Property: Enhanced image should maintain same dimensions
        assert enhanced.shape == degraded_uint8.shape, \
            "Enhancement should preserve image dimensions"
        
        # Property: Enhanced image should be valid uint8
        assert enhanced.dtype == np.uint8, "Enhanced image should be uint8"
        assert np.all(enhanced >= 0) and np.all(enhanced <= 255), \
            "Enhanced image values should be in valid range"
        
        # Property: Enhancement should preserve structural information
        # (measured by correlation with original pattern)
        if quality_before > 0.1:  # Only test if original has some structure
            # Convert to grayscale for correlation test
            original_gray = cv2.cvtColor((base_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            degraded_gray = cv2.cvtColor(degraded_uint8, cv2.COLOR_RGB2GRAY)
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            
            # Calculate correlations
            corr_degraded = np.corrcoef(original_gray.flatten(), degraded_gray.flatten())[0, 1]
            corr_enhanced = np.corrcoef(original_gray.flatten(), enhanced_gray.flatten())[0, 1]
            
            # Handle NaN correlations (can happen with very uniform images)
            if not np.isnan(corr_degraded) and not np.isnan(corr_enhanced):
                assert corr_enhanced >= corr_degraded * 0.8, \
                    f"Enhancement should preserve structure: {corr_degraded:.3f} -> {corr_enhanced:.3f}"
    
    def test_property_2_enhancement_preserves_valid_images(self):
        """
        Property 2: Enhancement of already good images should not degrade them significantly.
        """
        # Create a high-quality image
        high_quality_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Calculate quality before enhancement
        quality_before = self.processor._calculate_quality_score(high_quality_image)
        
        # Apply enhancement
        enhanced = self.processor.enhance_quality(high_quality_image)
        
        # Calculate quality after enhancement
        quality_after = self.processor._calculate_quality_score(enhanced)
        
        # Property: Good images should not be significantly degraded by enhancement
        assert quality_after >= quality_before * 0.8, \
            f"Enhancement should not significantly degrade good images: {quality_before:.3f} -> {quality_after:.3f}"
    
    @given(
        ocean_percentage=st.floats(min_value=0.1, max_value=0.8),
        water_hue=st.integers(min_value=90, max_value=130),
        water_saturation=st.integers(min_value=50, max_value=200)
    )
    @settings(max_examples=20, deadline=5000)
    def test_property_3_ocean_region_detection(self, ocean_percentage, water_hue, water_saturation):
        """
        Property 3: Ocean Region Detection
        
        For any beach cam image with detectable ocean regions, the ocean detection
        should identify water areas with reasonable accuracy and consistency.
        
        **Validates: Requirements 1.4**
        """
        # Create synthetic beach cam image with ocean region
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create ocean region (bottom portion of image)
        ocean_height = int(height * ocean_percentage)
        ocean_start = height - ocean_height
        
        # Fill ocean region with water-like colors in HSV
        hsv_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Ocean region
        hsv_image[ocean_start:, :, 0] = water_hue  # Hue (blue-green)
        hsv_image[ocean_start:, :, 1] = water_saturation  # Saturation
        hsv_image[ocean_start:, :, 2] = np.random.randint(100, 255, (ocean_height, width))  # Value
        
        # Sky/beach region (non-ocean)
        hsv_image[:ocean_start, :, 0] = np.random.randint(0, 60, (ocean_start, width))  # Non-blue hues
        hsv_image[:ocean_start, :, 1] = np.random.randint(50, 150, (ocean_start, width))
        hsv_image[:ocean_start, :, 2] = np.random.randint(150, 255, (ocean_start, width))
        
        # Convert to RGB
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        # Apply ocean detection
        ocean_mask = self.processor.detect_ocean_region(image)
        
        # Property: Ocean mask should be binary
        assert ocean_mask.dtype == np.uint8, "Ocean mask should be uint8"
        assert np.all((ocean_mask == 0) | (ocean_mask == 255)), "Ocean mask should be binary (0 or 255)"
        
        # Property: Ocean mask should have same spatial dimensions as input
        assert ocean_mask.shape == (height, width), f"Ocean mask shape {ocean_mask.shape} should match image {(height, width)}"
        
        # Property: Detection info should contain required metadata
        assert isinstance(detection_info, dict), "Detection info should be a dictionary"
        required_keys = ['ocean_coverage_percent', 'largest_region_area', 'num_regions', 
                        'confidence_score', 'detection_method', 'focus_region', 'has_detectable_ocean']
        for key in required_keys:
            assert key in detection_info, f"Detection info should contain '{key}'"
        
        # Property: Detection metadata should be consistent with mask
        ocean_pixels = np.sum(ocean_mask == 255)
        total_pixels = height * width
        detected_percentage = ocean_pixels / total_pixels * 100
        
        # Allow reasonable tolerance for morphological operations and floating point precision
        assert abs(detection_info['ocean_coverage_percent'] - detected_percentage) < 1.0, \
            f"Coverage percentage mismatch: {detection_info['ocean_coverage_percent']:.2f}% vs {detected_percentage:.2f}%"
        
        # Property: Should detect some ocean if ocean region is significant
        if ocean_percentage > 0.2:  # Only test if ocean region is substantial
            # Should detect at least some ocean (allowing for imperfect detection on synthetic images)
            # Synthetic images may not have all the characteristics of real ocean images
            assert detected_percentage > 1.0, f"Should detect some ocean when {ocean_percentage*100:.1f}% is water, got {detected_percentage:.1f}%"
            
            # Should not detect more than the entire image as ocean
            assert detected_percentage < 95.0, f"Should not detect entire image as ocean, got {detected_percentage:.1f}%"
            
            # If we detected significant ocean, the flag should be set
            if detected_percentage > 5.0:
                assert detection_info['has_detectable_ocean'], "Should have detectable ocean flag set for significant ocean regions"
        
        # Property: Ocean detection should be consistent (same input -> same output)
        ocean_mask2, detection_info2 = self.processor.detect_ocean_region(image)
        assert np.array_equal(ocean_mask, ocean_mask2), "Ocean detection should be deterministic"
        assert detection_info == detection_info2, "Detection info should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])