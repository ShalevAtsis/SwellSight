"""
Data augmentation with constraints for wave analysis.

Implements weather effect augmentations while preserving geometric scale
relationships needed for accurate height measurement.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from dataclasses import dataclass
import random
import logging

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation parameters."""
    enable_weather_effects: bool = True
    enable_lighting_changes: bool = True
    preserve_scale: bool = True  # Critical for height measurement
    max_brightness_change: float = 0.2
    max_contrast_change: float = 0.2
    fog_probability: float = 0.1
    rain_probability: float = 0.1
    glare_probability: float = 0.1
    
    # Weather effect parameters
    fog_intensity_range: Tuple[float, float] = (0.1, 0.4)
    rain_intensity_range: Tuple[float, float] = (0.05, 0.2)
    glare_intensity_range: Tuple[float, float] = (0.1, 0.3)
    
    # Scale preservation validation
    scale_tolerance: float = 0.05  # 5% tolerance for scale preservation
    edge_preservation_threshold: float = 0.8

class WaveAugmentation:
    """Constrained augmentation for wave analysis training."""
    
    def __init__(self, config: AugmentationConfig = None):
        """Initialize wave augmentation.
        
        Args:
            config: Augmentation configuration parameters
        """
        self.config = config or AugmentationConfig()
        self.logger = logging.getLogger(__name__)
        
    def apply_weather_effects(self, image: np.ndarray) -> np.ndarray:
        """Apply weather effect augmentations.
        
        Args:
            image: Input RGB image (H, W, 3)
            
        Returns:
            Augmented image with weather effects
        """
        augmented = image.copy().astype(np.float32) / 255.0
        
        # Apply fog effect
        if random.random() < self.config.fog_probability:
            augmented = self._apply_fog(augmented)
        
        # Apply rain effect
        if random.random() < self.config.rain_probability:
            augmented = self._apply_rain(augmented)
        
        # Apply glare effect
        if random.random() < self.config.glare_probability:
            augmented = self._apply_glare(augmented)
        
        # Convert back to uint8
        augmented = np.clip(augmented * 255.0, 0, 255).astype(np.uint8)
        return augmented
    
    def _apply_fog(self, image: np.ndarray) -> np.ndarray:
        """Apply fog effect to image.
        
        Args:
            image: Input image in [0, 1] range
            
        Returns:
            Image with fog effect applied
        """
        intensity = random.uniform(*self.config.fog_intensity_range)
        
        # Create fog mask - stronger at top (horizon), weaker at bottom
        h, w = image.shape[:2]
        fog_mask = np.linspace(intensity, intensity * 0.3, h)
        fog_mask = np.tile(fog_mask.reshape(-1, 1), (1, w))
        fog_mask = fog_mask[:, :, np.newaxis]
        
        # Apply fog as additive white noise with spatial variation
        fog_color = np.array([0.9, 0.9, 0.95])  # Slightly blue-tinted white
        fogged = image * (1 - fog_mask) + fog_color * fog_mask
        
        return np.clip(fogged, 0, 1)
    
    def _apply_rain(self, image: np.ndarray) -> np.ndarray:
        """Apply rain effect to image.
        
        Args:
            image: Input image in [0, 1] range
            
        Returns:
            Image with rain effect applied
        """
        intensity = random.uniform(*self.config.rain_intensity_range)
        h, w = image.shape[:2]
        
        # Create rain streaks
        num_drops = int(h * w * intensity * 0.001)
        rain_image = image.copy()
        
        for _ in range(num_drops):
            # Random rain drop position and length
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 10)
            length = random.randint(5, 15)
            
            # Draw rain streak
            end_y = min(y + length, h - 1)
            rain_image[y:end_y, x] = np.minimum(
                rain_image[y:end_y, x] + 0.3,
                1.0
            )
        
        # Add slight overall darkening
        rain_image = rain_image * (1 - intensity * 0.1)
        
        return np.clip(rain_image, 0, 1)
    
    def _apply_glare(self, image: np.ndarray) -> np.ndarray:
        """Apply sun glare effect to image.
        
        Args:
            image: Input image in [0, 1] range
            
        Returns:
            Image with glare effect applied
        """
        intensity = random.uniform(*self.config.glare_intensity_range)
        h, w = image.shape[:2]
        
        # Create glare center (usually in upper portion)
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(0, h // 3)
        
        # Create radial gradient for glare
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(h**2 + w**2) / 2
        
        # Normalize distance and create glare mask
        normalized_distance = distance / max_distance
        glare_mask = np.exp(-normalized_distance * 3) * intensity
        glare_mask = glare_mask[:, :, np.newaxis]
        
        # Apply glare as additive effect
        glare_color = np.array([1.0, 1.0, 0.8])  # Warm white
        glared = image + glare_mask * glare_color
        
        return np.clip(glared, 0, 1)
    
    def apply_lighting_changes(self, image: np.ndarray) -> np.ndarray:
        """Apply lighting variation augmentations.
        
        Args:
            image: Input RGB image
            
        Returns:
            Augmented image with lighting changes
        """
        augmented = image.copy().astype(np.float32)
        
        # Random brightness adjustment
        brightness_factor = 1.0 + random.uniform(
            -self.config.max_brightness_change,
            self.config.max_brightness_change
        )
        augmented = augmented * brightness_factor
        
        # Random contrast adjustment
        contrast_factor = 1.0 + random.uniform(
            -self.config.max_contrast_change,
            self.config.max_contrast_change
        )
        mean_value = np.mean(augmented)
        augmented = (augmented - mean_value) * contrast_factor + mean_value
        
        # Clip to valid range
        augmented = np.clip(augmented, 0, 255).astype(np.uint8)
        
        return augmented
    
    def validate_scale_preservation(self, 
                                  original: np.ndarray, 
                                  augmented: np.ndarray) -> bool:
        """Validate that geometric scale is preserved after augmentation.
        
        Args:
            original: Original image
            augmented: Augmented image
            
        Returns:
            True if scale relationships are preserved
        """
        try:
            # Convert to grayscale for edge detection
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            aug_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
            
            # Detect edges using Canny
            orig_edges = cv2.Canny(orig_gray, 50, 150)
            aug_edges = cv2.Canny(aug_gray, 50, 150)
            
            # Calculate edge preservation ratio
            orig_edge_count = np.sum(orig_edges > 0)
            aug_edge_count = np.sum(aug_edges > 0)
            
            if orig_edge_count == 0:
                return True  # No edges to preserve
            
            edge_preservation_ratio = aug_edge_count / orig_edge_count
            
            # Check if edge preservation is within acceptable range
            edge_preserved = (
                edge_preservation_ratio >= self.config.edge_preservation_threshold
            )
            
            # Additional check: compare structural similarity
            # Use template matching to check for major structural changes
            correlation = cv2.matchTemplate(orig_gray, aug_gray, cv2.TM_CCOEFF_NORMED)
            max_correlation = np.max(correlation)
            
            structure_preserved = max_correlation >= (1.0 - self.config.scale_tolerance)
            
            return edge_preserved and structure_preserved
            
        except Exception as e:
            self.logger.warning(f"Scale validation failed: {e}")
            return False
    
    def augment_training_sample(self, 
                               image: np.ndarray,
                               preserve_labels: bool = True) -> Dict[str, Any]:
        """Apply full augmentation pipeline to training sample.
        
        Args:
            image: Input training image
            preserve_labels: Whether to preserve label accuracy
            
        Returns:
            Dictionary with augmented image and metadata
        """
        original_image = image.copy()
        augmented_image = image.copy()
        applied_augmentations = []
        
        try:
            # Apply weather effects
            if self.config.enable_weather_effects:
                augmented_image = self.apply_weather_effects(augmented_image)
                applied_augmentations.append("weather_effects")
            
            # Apply lighting changes
            if self.config.enable_lighting_changes:
                augmented_image = self.apply_lighting_changes(augmented_image)
                applied_augmentations.append("lighting_changes")
            
            # Validate scale preservation for height measurement accuracy
            scale_preserved = True
            if preserve_labels and self.config.preserve_scale:
                scale_preserved = self.validate_scale_preservation(
                    original_image, augmented_image
                )
                
                if not scale_preserved:
                    self.logger.warning("Augmentation violated scale preservation constraint")
                    # Return original image if scale is not preserved
                    augmented_image = original_image
                    applied_augmentations = ["scale_violation_fallback"]
            
            return {
                "augmented_image": augmented_image,
                "applied_augmentations": applied_augmentations,
                "scale_preserved": scale_preserved,
                "augmentation_success": True
            }
            
        except Exception as e:
            self.logger.error(f"Augmentation failed: {e}")
            return {
                "augmented_image": original_image,
                "applied_augmentations": ["augmentation_failed"],
                "scale_preserved": True,  # Original image preserves scale
                "augmentation_success": False,
                "error": str(e)
            }
    
    def create_augmentation_variants(self, 
                                   image: np.ndarray, 
                                   num_variants: int = 3) -> List[Dict[str, Any]]:
        """Create multiple augmentation variants of an image.
        
        Args:
            image: Input image
            num_variants: Number of variants to create
            
        Returns:
            List of augmented image dictionaries
        """
        variants = []
        
        for i in range(num_variants):
            variant = self.augment_training_sample(image, preserve_labels=True)
            variant['variant_id'] = i
            variants.append(variant)
        
        return variants