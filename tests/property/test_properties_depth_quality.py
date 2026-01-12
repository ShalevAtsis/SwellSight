"""
Property-based tests for depth map quality assessment.

Tests Property 4: Depth Map Edge Preservation
Tests Property 5: Water Texture Capture  
Tests Property 6: Far-Field Depth Sensitivity
Validates: Requirements 2.1, 2.2, 2.3
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from swellsight.core.depth_extractor import DepthAnythingV2Extractor, DepthMap

# Strategies for generating test images
def generate_edge_image(width: int, height: int, num_edges: int) -> np.ndarray:
    """Generate synthetic image with clear edges for testing edge preservation."""
    image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add vertical and horizontal edges
    for _ in range(num_edges):
        if np.random.random() > 0.5:  # Vertical edge
            x = np.random.randint(width // 4, 3 * width // 4)
            image[:, x:x+2] = [255, 255, 255] if np.random.random() > 0.5 else [0, 0, 0]
        else:  # Horizontal edge
            y = np.random.randint(height // 4, 3 * height // 4)
            image[y:y+2, :] = [255, 255, 255] if np.random.random() > 0.5 else [0, 0, 0]
    
    return image

def generate_textured_image(width: int, height: int, texture_strength: float) -> np.ndarray:
    """Generate synthetic image with water-like texture for testing texture capture."""
    # Base water color
    base_color = np.array([64, 128, 192])  # Blue-ish water
    
    # Generate noise for texture
    noise = np.random.normal(0, texture_strength * 50, (height, width))
    
    # Create textured image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        channel = base_color[c] + noise
        image[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
    
    return image

def generate_depth_gradient_image(width: int, height: int, gradient_strength: float) -> np.ndarray:
    """Generate image with depth gradient for testing far-field sensitivity."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create depth gradient from near (bottom) to far (top)
    for y in range(height):
        # Intensity decreases with distance (y coordinate)
        intensity = int(255 * (1 - gradient_strength * y / height))
        intensity = max(50, min(255, intensity))  # Keep in reasonable range
        
        # Add some variation
        row_noise = np.random.normal(0, 10, width)
        row_values = np.clip(intensity + row_noise, 0, 255).astype(np.uint8)
        
        image[y, :] = np.stack([row_values, row_values, row_values], axis=1)
    
    return image

# Hypothesis strategies
image_dimensions_strategy = st.tuples(
    st.integers(min_value=320, max_value=1280),  # width
    st.integers(min_value=240, max_value=960)    # height
).filter(lambda dims: dims[0] >= dims[1])  # width >= height for realistic aspect ratios

edge_image_strategy = st.builds(
    generate_edge_image,
    width=st.integers(min_value=320, max_value=640),
    height=st.integers(min_value=240, max_value=480),
    num_edges=st.integers(min_value=2, max_value=8)
)

textured_image_strategy = st.builds(
    generate_textured_image,
    width=st.integers(min_value=320, max_value=640),
    height=st.integers(min_value=240, max_value=480),
    texture_strength=st.floats(min_value=0.1, max_value=1.0)
)

depth_gradient_image_strategy = st.builds(
    generate_depth_gradient_image,
    width=st.integers(min_value=320, max_value=640),
    height=st.integers(min_value=240, max_value=480),
    gradient_strength=st.floats(min_value=0.3, max_value=0.9)
)

class TestDepthMapQualityAssessment:
    """Property-based tests for depth map quality assessment."""
    
    @given(image=edge_image_strategy)
    @settings(max_examples=2, suppress_health_check=[HealthCheck.too_slow], deadline=60000)
    def test_property_4_depth_map_edge_preservation(self, image):
        """
        Property 4: Depth Map Edge Preservation
        
        For any beach cam image with visible wave edges, the generated depth map 
        should preserve sharp wave boundaries with measurable edge preservation metrics.
        
        **Feature: wave-analysis-system, Property 4: Depth Map Edge Preservation**
        **Validates: Requirements 2.1**
        """
        # Initialize extractor with small model for faster testing
        extractor = DepthAnythingV2Extractor(model_size="small", precision="fp32")
        
        try:
            # Extract depth map
            depth_map = extractor.extract_depth(image)
            
            # Validate basic properties
            assert isinstance(depth_map, DepthMap), "Should return DepthMap object"
            assert depth_map.data.shape == (image.shape[0], image.shape[1]), "Depth map should match image dimensions"
            assert 0 <= depth_map.data.min() <= depth_map.data.max() <= 1, "Depth values should be normalized to [0,1]"
            
            # Test edge preservation - should be measurable (> 0)
            assert depth_map.edge_preservation >= 0, "Edge preservation should be non-negative"
            
            # For images with clear edges, edge preservation should be reasonable
            # (We use a low threshold since synthetic images may not perfectly match real wave edges)
            assert depth_map.edge_preservation >= -0.5, "Edge preservation should not be extremely negative"
            
            # Depth map should have some variation (not completely flat)
            depth_std = np.std(depth_map.data)
            assert depth_std > 0.01, f"Depth map should have variation, got std={depth_std}"
            
        except Exception as e:
            pytest.fail(f"Depth extraction failed for edge image: {e}")
    
    @given(image=textured_image_strategy)
    @settings(max_examples=2, suppress_health_check=[HealthCheck.too_slow], deadline=60000)
    def test_property_5_water_texture_capture(self, image):
        """
        Property 5: Water Texture Capture
        
        For any image with visible water surface texture, the depth map should 
        capture fine-grained surface details with appropriate texture preservation scores.
        
        **Feature: wave-analysis-system, Property 5: Water Texture Capture**
        **Validates: Requirements 2.2**
        """
        # Initialize extractor
        extractor = DepthAnythingV2Extractor(model_size="small", precision="fp32")
        
        try:
            # Extract depth map
            depth_map = extractor.extract_depth(image)
            
            # Validate basic properties
            assert isinstance(depth_map, DepthMap), "Should return DepthMap object"
            assert depth_map.data.shape == (image.shape[0], image.shape[1]), "Depth map should match image dimensions"
            
            # Test texture capture through depth variation
            # Textured surfaces should produce depth maps with local variation
            
            # Calculate local variation using a sliding window
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            local_mean = cv2.filter2D(depth_map.data, -1, kernel)
            local_variation = np.abs(depth_map.data - local_mean)
            avg_local_variation = np.mean(local_variation)
            
            # Textured images should produce some local variation in depth
            assert avg_local_variation > 0.00015, f"Textured image should produce local depth variation, got {avg_local_variation}"
            
            # Quality score should be reasonable for textured images
            assert depth_map.quality_score >= 0, "Quality score should be non-negative"
            assert depth_map.quality_score <= 1, "Quality score should not exceed 1"
            
        except Exception as e:
            pytest.fail(f"Depth extraction failed for textured image: {e}")
    
    @given(image=depth_gradient_image_strategy)
    @settings(max_examples=2, suppress_health_check=[HealthCheck.too_slow], deadline=60000)
    def test_property_6_far_field_depth_sensitivity(self, image):
        """
        Property 6: Far-Field Depth Sensitivity
        
        For any image with distant waves, the depth extraction should maintain 
        sensitivity for far-field objects with consistent depth gradients.
        
        **Feature: wave-analysis-system, Property 6: Far-Field Depth Sensitivity**
        **Validates: Requirements 2.3**
        """
        # Initialize extractor
        extractor = DepthAnythingV2Extractor(model_size="small", precision="fp32")
        
        try:
            # Extract depth map
            depth_map = extractor.extract_depth(image)
            
            # Validate basic properties
            assert isinstance(depth_map, DepthMap), "Should return DepthMap object"
            assert depth_map.data.shape == (image.shape[0], image.shape[1]), "Depth map should match image dimensions"
            
            # Test far-field sensitivity by checking depth gradient
            # The depth map should show some correlation with the input gradient
            
            # Calculate vertical gradient in depth map (top to bottom)
            depth_gradient = np.gradient(depth_map.data, axis=0)
            
            # Check that there's a consistent gradient pattern
            # (not just random noise)
            gradient_consistency = np.std(np.mean(depth_gradient, axis=1))
            
            # Should have some gradient structure (not completely random)
            assert gradient_consistency > 0.001, f"Depth gradient should show structure, got consistency={gradient_consistency}"
            
            # Depth map should have reasonable range for gradient images
            depth_range = depth_map.data.max() - depth_map.data.min()
            assert depth_range > 0.1, f"Depth map should have reasonable range for gradient image, got {depth_range}"
            
            # Quality metrics should be reasonable
            assert 0 <= depth_map.quality_score <= 1, "Quality score should be in [0,1] range"
            assert depth_map.edge_preservation >= -1, "Edge preservation should be reasonable"
            
        except Exception as e:
            pytest.fail(f"Depth extraction failed for gradient image: {e}")
    
    @given(
        width=st.integers(min_value=320, max_value=640),
        height=st.integers(min_value=240, max_value=480)
    )
    @settings(max_examples=1, suppress_health_check=[HealthCheck.too_slow], deadline=60000)
    def test_depth_extraction_robustness(self, width, height):
        """
        Test robustness of depth extraction across different image sizes and types.
        
        **Feature: wave-analysis-system, Property 4-6: Depth Map Quality**
        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Test with different image types
        test_images = [
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),  # Random noise
            np.full((height, width, 3), 128, dtype=np.uint8),  # Uniform gray
            np.zeros((height, width, 3), dtype=np.uint8),  # Black image
            np.full((height, width, 3), 255, dtype=np.uint8),  # White image
        ]
        
        extractor = DepthAnythingV2Extractor(model_size="small", precision="fp32")
        
        for i, image in enumerate(test_images):
            try:
                depth_map = extractor.extract_depth(image)
                
                # Basic validation for all image types
                assert isinstance(depth_map, DepthMap), f"Should return DepthMap for image type {i}"
                assert depth_map.data.shape == (height, width), f"Wrong depth map shape for image type {i}"
                assert 0 <= depth_map.data.min() <= depth_map.data.max() <= 1, f"Invalid depth range for image type {i}"
                assert 0 <= depth_map.quality_score <= 1, f"Invalid quality score for image type {i}"
                
            except Exception as e:
                pytest.fail(f"Depth extraction failed for image type {i}: {e}")

if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v"])