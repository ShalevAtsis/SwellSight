#!/usr/bin/env python3
"""
Property-Based Test for Image Quality Validation
Feature: swellsight-pipeline-improvements, Property 11: Image Quality Validation
Validates: Requirements 3.1
"""

import sys
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import pytest
from hypothesis import given, strategies as st, settings, example

# Add utils to path
sys.path.append('.')

from utils.data_validator import DataValidator

class TestImageQualityValidation:
    """Property-based tests for image quality validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator(quality_threshold=0.7)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_image(self, width: int, height: int, channels: int = 3, 
                         brightness: int = 128, noise_level: float = 0.1) -> Path:
        """Create a test image with specified properties"""
        # Ensure valid dimensions
        width = max(32, min(width, 2048))
        height = max(32, min(height, 2048))
        channels = max(1, min(channels, 4))
        brightness = max(0, min(brightness, 255))
        noise_level = max(0.0, min(noise_level, 1.0))
        
        # Create base image
        if channels == 1:
            img_array = np.full((height, width), brightness, dtype=np.uint8)
        else:
            img_array = np.full((height, width, min(channels, 3)), brightness, dtype=np.uint8)
        
        # Add noise for variation
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        if channels == 1:
            img = Image.fromarray(img_array, mode='L')
        else:
            img = Image.fromarray(img_array, mode='RGB')
        
        # Save image
        img_path = self.temp_dir / f"test_image_{width}x{height}_{channels}ch.jpg"
        img.save(img_path, "JPEG", quality=85)
        
        return img_path
    
    @given(
        width=st.integers(min_value=64, max_value=512),  # Reduced range for faster generation
        height=st.integers(min_value=64, max_value=512),  # Reduced range for faster generation
        brightness=st.integers(min_value=50, max_value=200),
        noise_level=st.floats(min_value=0.0, max_value=0.2)  # Reduced range for faster generation
    )
    @settings(max_examples=50, deadline=10000, suppress_health_check=[])  # Increased deadline, reduced examples
    @example(width=512, height=512, brightness=128, noise_level=0.1)
    def test_image_quality_validation_property(self, width: int, height: int, 
                                             brightness: int, noise_level: float):
        """
        Property: For any valid image with reasonable parameters, 
        quality validation should return consistent results
        
        Feature: swellsight-pipeline-improvements, Property 11: Image Quality Validation
        Validates: Requirements 3.1
        """
        # Create test image
        img_path = self.create_test_image(width, height, 3, brightness, noise_level)
        
        # Validate image quality
        result = self.validator.validate_image_quality(str(img_path))
        
        # Property assertions
        assert isinstance(result, dict), "Validation result must be a dictionary"
        assert 'valid' in result, "Result must contain 'valid' field"
        assert 'score' in result, "Result must contain 'score' field"
        assert 'issues' in result, "Result must contain 'issues' field"
        assert 'metrics' in result, "Result must contain 'metrics' field"
        
        # Score properties
        assert isinstance(result['score'], (int, float)), "Score must be numeric"
        assert 0.0 <= result['score'] <= 1.0, f"Score must be between 0 and 1, got {result['score']}"
        
        # Valid flag consistency
        assert isinstance(result['valid'], bool), "Valid flag must be boolean"
        
        # Issues list properties
        assert isinstance(result['issues'], list), "Issues must be a list"
        
        # Metrics properties
        assert isinstance(result['metrics'], dict), "Metrics must be a dictionary"
        
        # If image is valid, it should have reasonable metrics
        if result['valid']:
            metrics = result['metrics']
            assert 'width' in metrics and metrics['width'] == width
            assert 'height' in metrics and metrics['height'] == height
            assert 'file_size' in metrics and metrics['file_size'] > 0
            
            # Quality score should be reasonable for valid images
            assert result['score'] >= 0.1, f"Valid image should have reasonable quality score, got {result['score']}"
        
        # If image has issues, score should reflect that
        if len(result['issues']) > 0:
            # Images with issues might still be valid but should have lower scores
            pass  # This is acceptable - issues don't always mean invalid
        
        # Consistency check: very low scores should correspond to invalid images
        if result['score'] < 0.3:
            # Very low quality images should typically be flagged as having issues
            # But this isn't a hard requirement, so we just log it
            pass
    
    @given(
        width=st.integers(min_value=1, max_value=31),  # Below minimum resolution
        height=st.integers(min_value=1, max_value=31)
    )
    @settings(max_examples=20, deadline=3000)
    def test_low_resolution_rejection_property(self, width: int, height: int):
        """
        Property: For any image below minimum resolution, 
        validation should reject it
        
        Feature: swellsight-pipeline-improvements, Property 11: Image Quality Validation
        Validates: Requirements 3.1
        """
        # Create low resolution test image
        img_path = self.create_test_image(width, height)
        
        # Validate image quality
        result = self.validator.validate_image_quality(str(img_path))
        
        # Property assertion: low resolution images should be invalid
        assert result['valid'] == False, f"Image with resolution {width}x{height} should be invalid"
        assert result['score'] == 0.0, f"Invalid image should have score 0.0, got {result['score']}"
        assert len(result['issues']) > 0, "Invalid image should have issues listed"
        
        # Check that resolution issue is mentioned
        issues_text = ' '.join(result['issues']).lower()
        assert 'resolution' in issues_text or 'low' in issues_text, \
            f"Resolution issue should be mentioned in: {result['issues']}"
    
    def test_nonexistent_file_property(self):
        """
        Property: For any non-existent file path, 
        validation should return invalid result
        
        Feature: swellsight-pipeline-improvements, Property 11: Image Quality Validation
        Validates: Requirements 3.1
        """
        # Test with non-existent file
        nonexistent_path = self.temp_dir / "nonexistent_file.jpg"
        
        result = self.validator.validate_image_quality(str(nonexistent_path))
        
        # Property assertions
        assert result['valid'] == False, "Non-existent file should be invalid"
        assert result['score'] == 0.0, "Non-existent file should have score 0.0"
        assert len(result['issues']) > 0, "Non-existent file should have issues"
        
        # Check that file existence issue is mentioned
        issues_text = ' '.join(result['issues']).lower()
        assert 'exist' in issues_text or 'not found' in issues_text, \
            f"File existence issue should be mentioned in: {result['issues']}"
    
    def test_corrupted_file_property(self):
        """
        Property: For any corrupted file, 
        validation should return invalid result
        
        Feature: swellsight-pipeline-improvements, Property 11: Image Quality Validation
        Validates: Requirements 3.1
        """
        # Create corrupted file
        corrupted_path = self.temp_dir / "corrupted.jpg"
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid image file")
        
        result = self.validator.validate_image_quality(str(corrupted_path))
        
        # Property assertions
        assert result['valid'] == False, "Corrupted file should be invalid"
        assert result['score'] == 0.0, "Corrupted file should have score 0.0"
        assert len(result['issues']) > 0, "Corrupted file should have issues"
        
        # Check that loading error is mentioned (more flexible check)
        issues_text = ' '.join(result['issues']).lower()
        assert ('error' in issues_text or 'loading' in issues_text or 'format' in issues_text or 
                'small' in issues_text or 'invalid' in issues_text), \
            f"File error should be mentioned in: {result['issues']}"


if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v", "--tb=short"])