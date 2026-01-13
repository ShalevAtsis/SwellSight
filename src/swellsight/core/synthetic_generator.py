"""
Stage B: Synthetic Data Factory

Generates diverse, labeled synthetic wave imagery using FLUX ControlNet for training
robust models without manual annotation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image
import logging
from pathlib import Path
from scipy import ndimage
from scipy.signal import find_peaks
import cv2

try:
    from diffusers import FluxControlNetPipeline, FluxControlNetModel
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers not available. FLUX ControlNet functionality will be limited.")

from .depth_extractor import DepthMap

@dataclass
class WeatherConditions:
    """Weather and lighting conditions for synthetic generation."""
    lighting: str  # "sunny", "overcast", "golden_hour", "stormy"
    weather: str   # "clear", "rain", "fog", "glare"
    wind_strength: float  # 0.0 to 1.0
    wave_foam: float      # 0.0 to 1.0

@dataclass
class GenerationConfig:
    """Configuration parameters for synthetic image generation."""
    resolution: tuple = (1024, 1024)
    controlnet_conditioning_scale: float = 0.5  # Range: 0.3-0.7 as per requirements
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    prompt: str = "Beautiful ocean waves breaking on a beach, photorealistic, high quality"
    negative_prompt: str = "blurry, low quality, distorted, artificial, cartoon"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0.3 <= self.controlnet_conditioning_scale <= 0.7):
            raise ValueError(
                f"controlnet_conditioning_scale must be between 0.3 and 0.7, "
                f"got {self.controlnet_conditioning_scale}"
            )
        if self.resolution != (1024, 1024):
            logging.warning(
                f"FLUX.1-dev is optimized for 1024x1024 resolution, "
                f"got {self.resolution}. This may affect quality."
            )

@dataclass
class WaveMetrics:
    """Ground truth wave metrics for synthetic images."""
    height_meters: float
    height_feet: float
    height_confidence: float
    direction: str  # "LEFT", "RIGHT", "STRAIGHT"
    direction_confidence: float
    breaking_type: str  # "SPILLING", "PLUNGING", "SURGING"
    breaking_confidence: float
    extreme_conditions: bool

@dataclass
class SyntheticImage:
    """Generated synthetic wave image with labels."""
    rgb_data: np.ndarray
    depth_map: DepthMap
    ground_truth_labels: WaveMetrics
    generation_params: GenerationConfig

@dataclass
class LabelValidationResult:
    """Result of label validation and consistency checking."""
    is_valid: bool
    confidence_score: float
    validation_errors: List[str]
    consistency_warnings: List[str]
    quality_metrics: Dict[str, float]

@dataclass
class WaveCharacteristics:
    """Detailed wave characteristics extracted from depth analysis."""
    peak_positions: List[Tuple[int, int]]  # (x, y) coordinates of wave peaks
    wave_count: int
    dominant_wavelength: float
    wave_steepness: float
    breaking_intensity: float
    foam_coverage: float
    depth_gradient_magnitude: float
    surface_roughness: float

class AutomaticLabelingSystem:
    """Advanced automatic labeling system for synthetic wave images."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the automatic labeling system.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Calibration parameters for wave height estimation
        self.height_calibration = {
            'depth_scale_factor': 8.5,  # Empirically determined scaling
            'min_height': 0.5,
            'max_height': 8.0,
            'baseline_offset': 0.1
        }
        
        # Direction analysis parameters
        self.direction_params = {
            'gradient_threshold': 0.008,
            'smoothing_kernel_size': 5,
            'edge_exclusion_ratio': 0.1  # Exclude edges from analysis
        }
        
        # Breaking type classification parameters
        self.breaking_params = {
            'spilling_wind_threshold': 0.6,
            'plunging_steepness_threshold': 0.12,
            'surging_gradient_threshold': 0.05
        }
    
    def extract_wave_characteristics(self, depth_map: DepthMap) -> WaveCharacteristics:
        """Extract detailed wave characteristics from depth map.
        
        Args:
            depth_map: Input depth map
            
        Returns:
            WaveCharacteristics with detailed analysis
        """
        depth_data = depth_map.data
        height, width = depth_data.shape
        
        # Apply Gaussian smoothing to reduce noise
        smoothed_depth = ndimage.gaussian_filter(depth_data, sigma=1.0)
        
        # Find wave peaks using local maxima detection
        # Convert to 8-bit for OpenCV processing
        depth_8bit = (smoothed_depth * 255).astype(np.uint8)
        
        # Use morphological operations to find peaks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(depth_8bit, cv2.MORPH_TOPHAT, kernel)
        
        # Find contours of wave peaks
        contours, _ = cv2.findContours(tophat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract peak positions
        peak_positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small noise
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    peak_positions.append((cx, cy))
        
        wave_count = len(peak_positions)
        
        # Calculate dominant wavelength using FFT
        # Take horizontal profile through center
        center_profile = smoothed_depth[height // 2, :]
        fft = np.fft.fft(center_profile)
        freqs = np.fft.fftfreq(len(center_profile))
        
        # Find dominant frequency (excluding DC component)
        power_spectrum = np.abs(fft[1:len(fft)//2])
        if len(power_spectrum) > 0:
            dominant_freq_idx = np.argmax(power_spectrum) + 1
            dominant_wavelength = 1.0 / abs(freqs[dominant_freq_idx]) if freqs[dominant_freq_idx] != 0 else width
        else:
            dominant_wavelength = width / 2.0
        
        # Calculate wave steepness (height/wavelength ratio)
        depth_range = np.max(smoothed_depth) - np.min(smoothed_depth)
        wave_steepness = depth_range / dominant_wavelength if dominant_wavelength > 0 else 0.0
        
        # Estimate breaking intensity from depth gradients
        grad_x = np.gradient(smoothed_depth, axis=1)
        grad_y = np.gradient(smoothed_depth, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        breaking_intensity = np.mean(gradient_magnitude)
        
        # Estimate foam coverage from high-gradient regions
        high_gradient_mask = gradient_magnitude > np.percentile(gradient_magnitude, 85)
        foam_coverage = np.sum(high_gradient_mask) / (height * width)
        
        # Calculate surface roughness
        surface_roughness = np.std(gradient_magnitude)
        
        return WaveCharacteristics(
            peak_positions=peak_positions,
            wave_count=wave_count,
            dominant_wavelength=dominant_wavelength,
            wave_steepness=wave_steepness,
            breaking_intensity=breaking_intensity,
            foam_coverage=foam_coverage,
            depth_gradient_magnitude=np.mean(gradient_magnitude),
            surface_roughness=surface_roughness
        )
    
    def estimate_wave_height(
        self, 
        characteristics: WaveCharacteristics, 
        depth_map: DepthMap
    ) -> Tuple[float, float]:
        """Estimate wave height with improved accuracy.
        
        Args:
            characteristics: Wave characteristics
            depth_map: Input depth map
            
        Returns:
            Tuple of (height_meters, confidence)
        """
        depth_data = depth_map.data
        
        # Calculate depth range with outlier removal
        depth_flat = depth_data.flatten()
        q1, q3 = np.percentile(depth_flat, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        filtered_depth = depth_flat[(depth_flat >= lower_bound) & (depth_flat <= upper_bound)]
        depth_range = np.max(filtered_depth) - np.min(filtered_depth)
        
        # Apply calibrated scaling
        raw_height = depth_range * self.height_calibration['depth_scale_factor']
        
        # Apply baseline offset and constraints
        height_meters = max(
            self.height_calibration['min_height'],
            min(self.height_calibration['max_height'], 
                raw_height + self.height_calibration['baseline_offset'])
        )
        
        # Calculate confidence based on wave characteristics
        confidence = 0.95  # Base confidence for synthetic data
        
        # Adjust confidence based on wave count and steepness
        if characteristics.wave_count < 2:
            confidence *= 0.9  # Lower confidence for single waves
        elif characteristics.wave_count > 10:
            confidence *= 0.85  # Lower confidence for very complex scenes
            
        if characteristics.wave_steepness > 0.2:
            confidence *= 0.9  # Lower confidence for very steep waves
            
        return height_meters, confidence
    
    def classify_wave_direction(
        self, 
        characteristics: WaveCharacteristics, 
        depth_map: DepthMap
    ) -> Tuple[str, float]:
        """Classify wave breaking direction.
        
        Args:
            characteristics: Wave characteristics
            depth_map: Input depth map
            
        Returns:
            Tuple of (direction, confidence)
        """
        depth_data = depth_map.data
        height, width = depth_data.shape
        
        # Exclude edges from analysis
        edge_margin = int(width * self.direction_params['edge_exclusion_ratio'])
        analysis_region = depth_data[:, edge_margin:width-edge_margin]
        
        # Apply smoothing
        kernel_size = self.direction_params['smoothing_kernel_size']
        smoothed = ndimage.uniform_filter(analysis_region, size=kernel_size)
        
        # Calculate horizontal gradient
        grad_x = np.gradient(smoothed, axis=1)
        mean_gradient = np.mean(grad_x)
        
        # Classify direction based on gradient
        threshold = self.direction_params['gradient_threshold']
        
        if mean_gradient > threshold:
            direction = "RIGHT"
            confidence = min(0.95, 0.7 + abs(mean_gradient) * 20)
        elif mean_gradient < -threshold:
            direction = "LEFT"
            confidence = min(0.95, 0.7 + abs(mean_gradient) * 20)
        else:
            direction = "STRAIGHT"
            confidence = 0.85  # Slightly lower confidence for straight waves
            
        # Adjust confidence based on wave characteristics
        if characteristics.wave_count > 5:
            confidence *= 0.9  # Multiple waves may have mixed directions
            
        return direction, confidence
    
    def classify_breaking_type(
        self, 
        characteristics: WaveCharacteristics, 
        conditions: WeatherConditions
    ) -> Tuple[str, float]:
        """Classify wave breaking type.
        
        Args:
            characteristics: Wave characteristics
            conditions: Weather conditions
            
        Returns:
            Tuple of (breaking_type, confidence)
        """
        # Primary classification based on wind strength
        if conditions.wind_strength > self.breaking_params['spilling_wind_threshold']:
            breaking_type = "SPILLING"
            confidence = 0.9
        elif characteristics.wave_steepness > self.breaking_params['plunging_steepness_threshold']:
            breaking_type = "PLUNGING"
            confidence = 0.88
        elif characteristics.depth_gradient_magnitude < self.breaking_params['surging_gradient_threshold']:
            breaking_type = "SURGING"
            confidence = 0.85
        else:
            # Default to plunging for moderate conditions
            breaking_type = "PLUNGING"
            confidence = 0.82
            
        # Adjust confidence based on foam coverage
        if characteristics.foam_coverage > 0.3:
            if breaking_type == "SPILLING":
                confidence = min(0.95, confidence + 0.05)  # High foam supports spilling
            else:
                confidence *= 0.95  # High foam with non-spilling is less certain
                
        return breaking_type, confidence
    
    def validate_labels(
        self, 
        wave_metrics: WaveMetrics, 
        characteristics: WaveCharacteristics,
        conditions: WeatherConditions
    ) -> LabelValidationResult:
        """Validate generated labels for consistency and realism.
        
        Args:
            wave_metrics: Generated wave metrics
            characteristics: Wave characteristics
            conditions: Weather conditions
            
        Returns:
            LabelValidationResult with validation details
        """
        validation_errors = []
        consistency_warnings = []
        quality_metrics = {}
        
        # Validate wave height
        if not (0.5 <= wave_metrics.height_meters <= 8.0):
            validation_errors.append(f"Wave height {wave_metrics.height_meters}m outside valid range [0.5, 8.0]")
            
        # Check height-feet conversion
        expected_feet = wave_metrics.height_meters * 3.28084
        if abs(wave_metrics.height_feet - expected_feet) > 0.01:
            validation_errors.append("Height conversion meters to feet is incorrect")
            
        # Validate direction consistency
        valid_directions = ["LEFT", "RIGHT", "STRAIGHT"]
        if wave_metrics.direction not in valid_directions:
            validation_errors.append(f"Invalid direction: {wave_metrics.direction}")
            
        # Validate breaking type
        valid_breaking_types = ["SPILLING", "PLUNGING", "SURGING"]
        if wave_metrics.breaking_type not in valid_breaking_types:
            validation_errors.append(f"Invalid breaking type: {wave_metrics.breaking_type}")
            
        # Check confidence scores
        confidence_scores = [
            wave_metrics.height_confidence,
            wave_metrics.direction_confidence,
            wave_metrics.breaking_confidence
        ]
        
        for i, conf in enumerate(confidence_scores):
            if not (0.0 <= conf <= 1.0):
                validation_errors.append(f"Confidence score {i} outside valid range [0.0, 1.0]")
                
        # Consistency checks
        if conditions.wind_strength > 0.7 and wave_metrics.breaking_type != "SPILLING":
            consistency_warnings.append("High wind conditions typically produce spilling waves")
            
        if wave_metrics.height_meters > 5.0 and not wave_metrics.extreme_conditions:
            consistency_warnings.append("Large waves should be flagged as extreme conditions")
            
        if characteristics.wave_steepness > 0.15 and wave_metrics.breaking_type == "SURGING":
            consistency_warnings.append("Steep waves rarely produce surging breaks")
            
        # Calculate quality metrics
        quality_metrics['wave_count'] = characteristics.wave_count
        quality_metrics['steepness'] = characteristics.wave_steepness
        quality_metrics['breaking_intensity'] = characteristics.breaking_intensity
        quality_metrics['foam_coverage'] = characteristics.foam_coverage
        
        # Overall confidence score
        base_confidence = np.mean(confidence_scores)
        error_penalty = len(validation_errors) * 0.2
        warning_penalty = len(consistency_warnings) * 0.05
        
        overall_confidence = max(0.0, base_confidence - error_penalty - warning_penalty)
        
        is_valid = len(validation_errors) == 0
        
        return LabelValidationResult(
            is_valid=is_valid,
            confidence_score=overall_confidence,
            validation_errors=validation_errors,
            consistency_warnings=consistency_warnings,
            quality_metrics=quality_metrics
        )

@dataclass
@dataclass
class LabeledDataset:
    """Collection of labeled synthetic images."""
    images: List[SyntheticImage]
    balance_metrics: Dict[str, Dict[str, int]]
    statistics: Dict[str, Any]

class SyntheticDataGenerator(ABC):
    """Abstract base class for synthetic data generation."""
    
    @abstractmethod
    def generate_wave_scene(
        self, 
        depth_map: DepthMap, 
        conditions: WeatherConditions,
        config: Optional[GenerationConfig] = None
    ) -> SyntheticImage:
        """Generate synthetic wave image conditioned on depth map.
        
        Args:
            depth_map: Input depth map for conditioning
            conditions: Weather and lighting conditions
            config: Generation configuration (optional)
            
        Returns:
            SyntheticImage with generated RGB data and labels
        """
        pass
    
    @abstractmethod
    def create_balanced_dataset(self, target_size: int) -> LabeledDataset:
        """Generate balanced dataset across all wave conditions.
        
        Args:
            target_size: Number of synthetic images to generate
            
        Returns:
            LabeledDataset with balanced wave conditions
        """
        pass


def create_default_weather_conditions() -> List[WeatherConditions]:
    """Create a diverse set of default weather conditions for generation.
    
    Returns:
        List of WeatherConditions covering various scenarios
    """
    conditions = []
    
    # Sunny conditions
    conditions.extend([
        WeatherConditions("sunny", "clear", 0.2, 0.3),
        WeatherConditions("sunny", "clear", 0.5, 0.6),
        WeatherConditions("sunny", "glare", 0.3, 0.4),
    ])
    
    # Overcast conditions
    conditions.extend([
        WeatherConditions("overcast", "clear", 0.4, 0.5),
        WeatherConditions("overcast", "fog", 0.2, 0.2),
        WeatherConditions("overcast", "rain", 0.6, 0.7),
    ])
    
    # Golden hour conditions
    conditions.extend([
        WeatherConditions("golden_hour", "clear", 0.3, 0.4),
        WeatherConditions("golden_hour", "clear", 0.1, 0.2),
    ])
    
    # Stormy conditions
    conditions.extend([
        WeatherConditions("stormy", "rain", 0.8, 0.9),
        WeatherConditions("stormy", "clear", 0.7, 0.8),
    ])
    
    return conditions


def validate_generation_config(config: GenerationConfig) -> None:
    """Validate generation configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.resolution[0] <= 0 or config.resolution[1] <= 0:
        raise ValueError(f"Invalid resolution: {config.resolution}")
        
    if not (0.0 <= config.controlnet_conditioning_scale <= 1.0):
        raise ValueError(
            f"controlnet_conditioning_scale must be between 0.0 and 1.0, "
            f"got {config.controlnet_conditioning_scale}"
        )
        
    if config.num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {config.num_inference_steps}")
        
    if config.guidance_scale < 0:
        raise ValueError(f"guidance_scale must be non-negative, got {config.guidance_scale}")


def estimate_generation_time(
    config: GenerationConfig, 
    device: str = "cuda"
) -> float:
    """Estimate generation time for a single image.
    
    Args:
        config: Generation configuration
        device: Target device ("cuda" or "cpu")
        
    Returns:
        Estimated generation time in seconds
    """
    # Base time estimates (rough approximations)
    base_times = {
        "cuda": 15.0,  # seconds for GPU
        "cpu": 120.0   # seconds for CPU
    }
    
    base_time = base_times.get(device, base_times["cpu"])
    
    # Scale by inference steps
    step_factor = config.num_inference_steps / 50.0
    
    # Scale by resolution
    pixel_count = config.resolution[0] * config.resolution[1]
    resolution_factor = pixel_count / (1024 * 1024)
    
    estimated_time = base_time * step_factor * resolution_factor
    
    return estimated_time

class FLUXControlNetGenerator(SyntheticDataGenerator):
    """FLUX.1-dev + ControlNet implementation for synthetic wave generation."""
    
    def __init__(
        self, 
        model_path: str = "black-forest-labs/FLUX.1-dev",
        controlnet_path: str = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        """Initialize FLUX ControlNet generator.
        
        Args:
            model_path: Hugging Face model path for FLUX.1-dev
            controlnet_path: Hugging Face model path for ControlNet-Depth
            device: Device to run inference on (auto-detected if None)
            torch_dtype: Torch data type for inference (float16 for efficiency)
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "Diffusers library is required for FLUX ControlNet generation. "
                "Install with: pip install diffusers>=0.20.0"
            )
            
        self.model_path = model_path
        self.controlnet_path = controlnet_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self._pipeline = None
        self._controlnet = None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize automatic labeling system
        self.labeling_system = AutomaticLabelingSystem(self.logger)
        
    def _load_pipeline(self) -> None:
        """Load FLUX ControlNet pipeline with proper configuration."""
        if self._pipeline is not None:
            return
            
        try:
            self.logger.info(f"Loading FLUX ControlNet from {self.controlnet_path}")
            
            # Load ControlNet model
            self._controlnet = FluxControlNetModel.from_pretrained(
                self.controlnet_path,
                torch_dtype=self.torch_dtype
            )
            
            # Load FLUX pipeline with ControlNet
            self._pipeline = FluxControlNetPipeline.from_pretrained(
                self.model_path,
                controlnet=self._controlnet,
                torch_dtype=self.torch_dtype
            )
            
            # Move to device
            self._pipeline = self._pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self._pipeline, "enable_attention_slicing"):
                self._pipeline.enable_attention_slicing()
                
            if hasattr(self._pipeline, "enable_model_cpu_offload"):
                self._pipeline.enable_model_cpu_offload()
                
            self.logger.info(f"FLUX ControlNet pipeline loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load FLUX ControlNet pipeline: {e}")
            raise RuntimeError(f"Pipeline loading failed: {e}")
    
    def _prepare_depth_image(self, depth_map: DepthMap) -> Image.Image:
        """Prepare depth map for ControlNet conditioning.
        
        Args:
            depth_map: Input depth map
            
        Returns:
            PIL Image formatted for ControlNet
        """
        # Normalize depth data to 0-255 range
        depth_normalized = (depth_map.data * 255).astype(np.uint8)
        
        # Convert to 3-channel image (ControlNet expects RGB format)
        if len(depth_normalized.shape) == 2:
            depth_rgb = np.stack([depth_normalized] * 3, axis=-1)
        else:
            depth_rgb = depth_normalized
            
        # Convert to PIL Image
        depth_image = Image.fromarray(depth_rgb)
        
        # Resize to target resolution if needed
        target_size = (1024, 1024)  # FLUX.1-dev optimal resolution
        if depth_image.size != target_size:
            depth_image = depth_image.resize(target_size, Image.Resampling.LANCZOS)
            
        return depth_image
    
    def _create_weather_prompt(self, conditions: WeatherConditions, base_prompt: str) -> str:
        """Create weather-conditioned prompt for generation.
        
        Args:
            conditions: Weather and lighting conditions
            base_prompt: Base prompt for wave generation
            
        Returns:
            Enhanced prompt with weather conditions
        """
        # Build weather descriptors
        weather_descriptors = []
        
        # Lighting conditions
        lighting_map = {
            "sunny": "bright sunny day, clear blue sky",
            "overcast": "overcast sky, soft diffused lighting",
            "golden_hour": "golden hour lighting, warm sunset glow",
            "stormy": "dramatic stormy sky, dark clouds"
        }
        weather_descriptors.append(lighting_map.get(conditions.lighting, "natural lighting"))
        
        # Weather effects
        weather_map = {
            "clear": "clear weather",
            "rain": "light rain, wet surfaces",
            "fog": "misty fog, reduced visibility",
            "glare": "bright sun glare, high contrast"
        }
        weather_descriptors.append(weather_map.get(conditions.weather, "clear conditions"))
        
        # Wind and foam effects
        if conditions.wind_strength > 0.7:
            weather_descriptors.append("strong winds, choppy water")
        elif conditions.wind_strength > 0.3:
            weather_descriptors.append("moderate breeze")
            
        if conditions.wave_foam > 0.5:
            weather_descriptors.append("white foam, breaking waves")
            
        # Combine with base prompt
        enhanced_prompt = f"{base_prompt}, {', '.join(weather_descriptors)}"
        return enhanced_prompt
    
    def _extract_wave_metrics(
        self, 
        depth_map: DepthMap, 
        conditions: WeatherConditions,
        generation_config: GenerationConfig
    ) -> WaveMetrics:
        """Extract ground truth wave metrics using the automatic labeling system.
        
        Args:
            depth_map: Input depth map
            conditions: Generation conditions
            generation_config: Generation configuration
            
        Returns:
            WaveMetrics with automatically generated ground truth values
        """
        try:
            # Extract detailed wave characteristics
            characteristics = self.labeling_system.extract_wave_characteristics(depth_map)
            
            # Estimate wave height with improved accuracy
            height_meters, height_confidence = self.labeling_system.estimate_wave_height(
                characteristics, depth_map
            )
            height_feet = height_meters * 3.28084
            
            # Classify wave direction
            direction, direction_confidence = self.labeling_system.classify_wave_direction(
                characteristics, depth_map
            )
            
            # Classify breaking type
            breaking_type, breaking_confidence = self.labeling_system.classify_breaking_type(
                characteristics, conditions
            )
            
            # Check for extreme conditions
            extreme_conditions = height_meters < 0.8 or height_meters > 6.0
            
            # Create wave metrics
            wave_metrics = WaveMetrics(
                height_meters=height_meters,
                height_feet=height_feet,
                height_confidence=height_confidence,
                direction=direction,
                direction_confidence=direction_confidence,
                breaking_type=breaking_type,
                breaking_confidence=breaking_confidence,
                extreme_conditions=extreme_conditions
            )
            
            # Validate labels for consistency
            validation_result = self.labeling_system.validate_labels(
                wave_metrics, characteristics, conditions
            )
            
            # Log validation results
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Label validation failed: {validation_result.validation_errors}"
                )
            
            if validation_result.consistency_warnings:
                self.logger.info(
                    f"Label consistency warnings: {validation_result.consistency_warnings}"
                )
                
            self.logger.debug(
                f"Generated labels - Height: {height_meters:.2f}m, "
                f"Direction: {direction}, Breaking: {breaking_type}, "
                f"Validation confidence: {validation_result.confidence_score:.3f}"
            )
            
            return wave_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to extract wave metrics: {e}")
            # Fallback to basic extraction
            return self._extract_wave_metrics_fallback(depth_map, conditions)
    
    def validate_generated_labels(
        self,
        synthetic_image: SyntheticImage,
        conditions: WeatherConditions
    ) -> LabelValidationResult:
        """Validate labels of a generated synthetic image.
        
        Args:
            synthetic_image: Generated synthetic image with labels
            conditions: Weather conditions used for generation
            
        Returns:
            LabelValidationResult with validation details
        """
        try:
            # Extract characteristics from the depth map
            characteristics = self.labeling_system.extract_wave_characteristics(
                synthetic_image.depth_map
            )
            
            # Validate the labels
            validation_result = self.labeling_system.validate_labels(
                synthetic_image.ground_truth_labels,
                characteristics,
                conditions
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate labels: {e}")
            return LabelValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_errors=[f"Validation failed: {str(e)}"],
                consistency_warnings=[],
                quality_metrics={}
            )
    
    def _extract_wave_metrics_fallback(
        self, 
        depth_map: DepthMap, 
        conditions: WeatherConditions
    ) -> WaveMetrics:
        """Fallback method for wave metrics extraction if advanced method fails.
        
        Args:
            depth_map: Input depth map
            conditions: Generation conditions
            
        Returns:
            WaveMetrics with basic estimation
        """
        depth_data = depth_map.data
        
        # Basic height estimation
        depth_range = np.max(depth_data) - np.min(depth_data)
        height_meters = max(0.5, min(8.0, depth_range * 8.0))
        height_feet = height_meters * 3.28084
        
        # Basic direction estimation
        grad_x = np.gradient(depth_data, axis=1)
        mean_grad_x = np.mean(grad_x)
        
        if mean_grad_x > 0.01:
            direction = "RIGHT"
        elif mean_grad_x < -0.01:
            direction = "LEFT"
        else:
            direction = "STRAIGHT"
            
        # Basic breaking type estimation
        if conditions.wind_strength > 0.6:
            breaking_type = "SPILLING"
        elif depth_range > 0.12:
            breaking_type = "PLUNGING"
        else:
            breaking_type = "SURGING"
            
        # Lower confidence for fallback method
        height_confidence = 0.8
        direction_confidence = 0.75
        breaking_confidence = 0.7
        
        extreme_conditions = height_meters < 0.8 or height_meters > 6.0
        
        return WaveMetrics(
            height_meters=height_meters,
            height_feet=height_feet,
            height_confidence=height_confidence,
            direction=direction,
            direction_confidence=direction_confidence,
            breaking_type=breaking_type,
            breaking_confidence=breaking_confidence,
            extreme_conditions=extreme_conditions
        )
        
    def generate_wave_scene(
        self, 
        depth_map: DepthMap, 
        conditions: WeatherConditions,
        config: Optional[GenerationConfig] = None
    ) -> SyntheticImage:
        """Generate synthetic wave scene using FLUX ControlNet.
        
        Args:
            depth_map: Input depth map for conditioning
            conditions: Weather and lighting conditions
            config: Generation configuration (uses default if None)
            
        Returns:
            SyntheticImage with generated RGB data and labels
        """
        # Use default config if none provided
        if config is None:
            config = GenerationConfig()
            
        # Load pipeline if not already loaded
        self._load_pipeline()
        
        try:
            # Prepare depth image for ControlNet
            depth_image = self._prepare_depth_image(depth_map)
            
            # Create weather-conditioned prompt
            prompt = self._create_weather_prompt(conditions, config.prompt)
            
            # Set random seed if provided
            generator = None
            if config.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(config.seed)
                
            self.logger.info(f"Generating synthetic wave scene with prompt: {prompt[:100]}...")
            
            # Generate image using FLUX ControlNet
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=config.negative_prompt,
                control_image=depth_image,
                controlnet_conditioning_scale=config.controlnet_conditioning_scale,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=generator,
                height=config.resolution[1],
                width=config.resolution[0],
                return_dict=True
            )
            
            # Convert PIL image to numpy array
            generated_image = result.images[0]
            rgb_data = np.array(generated_image)
            
            # Extract ground truth labels
            ground_truth_labels = self._extract_wave_metrics(depth_map, conditions, config)
            
            self.logger.info("Synthetic wave scene generated successfully")
            
            return SyntheticImage(
                rgb_data=rgb_data,
                depth_map=depth_map,
                ground_truth_labels=ground_truth_labels,
                generation_params=config
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate synthetic wave scene: {e}")
            raise RuntimeError(f"Generation failed: {e}")
        
    def create_balanced_dataset(self, target_size: int) -> LabeledDataset:
        """Generate balanced dataset across all wave conditions.
        
        Args:
            target_size: Number of synthetic images to generate
            
        Returns:
            LabeledDataset with balanced wave conditions
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
            
        self.logger.info(f"Starting balanced dataset generation for {target_size} images")
        
        # Create diverse weather conditions for balanced generation
        weather_conditions = create_default_weather_conditions()
        
        # Define wave condition categories for balancing
        wave_categories = {
            'height_ranges': [
                (0.5, 1.5),   # Small waves
                (1.5, 3.0),   # Medium waves  
                (3.0, 5.0),   # Large waves
                (5.0, 8.0)    # Extra large waves
            ],
            'directions': ['LEFT', 'RIGHT', 'STRAIGHT'],
            'breaking_types': ['SPILLING', 'PLUNGING', 'SURGING']
        }
        
        # Calculate balanced distribution
        num_height_ranges = len(wave_categories['height_ranges'])
        num_directions = len(wave_categories['directions'])
        num_breaking_types = len(wave_categories['breaking_types'])
        
        # Distribute images across categories
        images_per_height = target_size // num_height_ranges
        remaining_images = target_size % num_height_ranges
        
        generated_images = []
        balance_metrics = {
            'height_distribution': {},
            'direction_distribution': {},
            'breaking_type_distribution': {},
            'weather_distribution': {}
        }
        
        # Initialize counters
        for height_range in wave_categories['height_ranges']:
            balance_metrics['height_distribution'][f"{height_range[0]}-{height_range[1]}m"] = 0
        for direction in wave_categories['directions']:
            balance_metrics['direction_distribution'][direction] = 0
        for breaking_type in wave_categories['breaking_types']:
            balance_metrics['breaking_type_distribution'][breaking_type] = 0
        for i, condition in enumerate(weather_conditions):
            balance_metrics['weather_distribution'][f"{condition.lighting}_{condition.weather}"] = 0
        
        try:
            # Generate images for each height category
            for height_idx, (min_height, max_height) in enumerate(wave_categories['height_ranges']):
                # Add extra image to first categories if there's remainder
                category_size = images_per_height + (1 if height_idx < remaining_images else 0)
                
                self.logger.info(f"Generating {category_size} images for height range {min_height}-{max_height}m")
                
                for img_idx in range(category_size):
                    try:
                        # Select weather conditions cyclically
                        weather_idx = (len(generated_images)) % len(weather_conditions)
                        conditions = weather_conditions[weather_idx]
                        
                        # Generate synthetic depth map targeting specific wave characteristics
                        depth_map = self._generate_target_depth_map(
                            target_height_range=(min_height, max_height),
                            target_direction=wave_categories['directions'][img_idx % num_directions],
                            target_breaking=wave_categories['breaking_types'][img_idx % num_breaking_types],
                            conditions=conditions
                        )
                        
                        # Generate synthetic image
                        config = GenerationConfig(
                            seed=42 + len(generated_images),  # Reproducible but varied
                            controlnet_conditioning_scale=0.3 + (0.4 * np.random.random())  # Vary within valid range
                        )
                        
                        synthetic_image = self.generate_wave_scene(depth_map, conditions, config)
                        
                        # Validate the generated labels
                        validation_result = self.validate_generated_labels(synthetic_image, conditions)
                        
                        if validation_result.is_valid and validation_result.confidence_score > 0.7:
                            generated_images.append(synthetic_image)
                            
                            # Update balance metrics
                            labels = synthetic_image.ground_truth_labels
                            
                            # Height distribution
                            for height_range in wave_categories['height_ranges']:
                                if height_range[0] <= labels.height_meters < height_range[1]:
                                    balance_metrics['height_distribution'][f"{height_range[0]}-{height_range[1]}m"] += 1
                                    break
                            
                            # Direction distribution
                            balance_metrics['direction_distribution'][labels.direction] += 1
                            
                            # Breaking type distribution
                            balance_metrics['breaking_type_distribution'][labels.breaking_type] += 1
                            
                            # Weather distribution
                            weather_key = f"{conditions.lighting}_{conditions.weather}"
                            balance_metrics['weather_distribution'][weather_key] += 1
                            
                            # Progress tracking
                            progress = len(generated_images) / target_size * 100
                            if len(generated_images) % max(1, target_size // 10) == 0:
                                self.logger.info(f"Progress: {progress:.1f}% ({len(generated_images)}/{target_size})")
                        else:
                            self.logger.warning(
                                f"Generated image failed validation (confidence: {validation_result.confidence_score:.3f}), "
                                f"errors: {validation_result.validation_errors}"
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Failed to generate image {img_idx} in height category {height_idx}: {e}")
                        continue
                        
            # Calculate final statistics
            statistics = self._calculate_dataset_statistics(generated_images, balance_metrics)
            
            # Validate dataset balance
            balance_validation = self._validate_dataset_balance(balance_metrics, target_size)
            
            self.logger.info(f"Generated {len(generated_images)} images successfully")
            self.logger.info(f"Dataset balance validation: {balance_validation}")
            
            return LabeledDataset(
                images=generated_images,
                balance_metrics=balance_metrics,
                statistics=statistics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create balanced dataset: {e}")
            raise RuntimeError(f"Balanced dataset generation failed: {e}")
    
    def _generate_target_depth_map(
        self,
        target_height_range: Tuple[float, float],
        target_direction: str,
        target_breaking: str,
        conditions: WeatherConditions,
        resolution: Tuple[int, int] = (256, 256)
    ) -> DepthMap:
        """Generate synthetic depth map with target wave characteristics.
        
        Args:
            target_height_range: Desired wave height range (min, max) in meters
            target_direction: Target wave direction ('LEFT', 'RIGHT', 'STRAIGHT')
            target_breaking: Target breaking type ('SPILLING', 'PLUNGING', 'SURGING')
            conditions: Weather conditions affecting wave generation
            resolution: Output resolution for depth map
            
        Returns:
            DepthMap with target characteristics
        """
        height, width = resolution
        
        # Create base ocean surface
        x = np.linspace(0, 2 * np.pi, width)
        y = np.linspace(0, 2 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate wave height within target range
        target_height = np.random.uniform(target_height_range[0], target_height_range[1])
        
        # Scale depth values based on target height
        # Use empirical scaling factor from labeling system
        depth_scale = target_height / 8.5  # Inverse of height_calibration['depth_scale_factor']
        
        # Create wave pattern based on direction
        if target_direction == "RIGHT":
            # Waves breaking from left to right
            wave_pattern = np.sin(2 * X + 0.5 * Y) * depth_scale
            # Add directional gradient
            gradient = np.linspace(-0.1, 0.1, width)
            wave_pattern += gradient[np.newaxis, :]
        elif target_direction == "LEFT":
            # Waves breaking from right to left  
            wave_pattern = np.sin(2 * X - 0.5 * Y) * depth_scale
            # Add directional gradient
            gradient = np.linspace(0.1, -0.1, width)
            wave_pattern += gradient[np.newaxis, :]
        else:  # STRAIGHT
            # Waves breaking straight
            wave_pattern = np.sin(2 * X) * depth_scale
            
        # Modify pattern based on breaking type
        if target_breaking == "SPILLING":
            # Add wind-driven texture for spilling waves
            wind_texture = np.random.normal(0, 0.02, (height, width)) * conditions.wind_strength
            wave_pattern += wind_texture
        elif target_breaking == "PLUNGING":
            # Add steeper gradients for plunging waves
            steep_factor = 1.5
            wave_pattern = np.sin(2 * X * steep_factor) * depth_scale
        elif target_breaking == "SURGING":
            # Smoother, less steep waves for surging
            smooth_factor = 0.7
            wave_pattern = np.sin(2 * X * smooth_factor) * depth_scale
            
        # Add foam effects based on conditions
        if conditions.wave_foam > 0.5:
            foam_mask = np.random.random((height, width)) < conditions.wave_foam * 0.3
            wave_pattern[foam_mask] += np.random.uniform(0.05, 0.15, np.sum(foam_mask))
            
        # Ensure depth values are in valid range [0, 1]
        wave_pattern = np.clip(wave_pattern + 0.5, 0.0, 1.0)
        
        # Apply Gaussian smoothing for realism
        from scipy import ndimage
        wave_pattern = ndimage.gaussian_filter(wave_pattern, sigma=1.0)
        
        # Calculate quality metrics
        gradients = np.gradient(wave_pattern)
        edge_preservation = np.std(gradients[0]) + np.std(gradients[1])  # Higher std = better edge preservation
        quality_score = min(1.0, edge_preservation * 2.0)  # Normalize to [0, 1]
        
        return DepthMap(
            data=wave_pattern.astype(np.float32),
            resolution=resolution,
            quality_score=quality_score,
            edge_preservation=edge_preservation
        )
    
    def _calculate_dataset_statistics(
        self, 
        images: List[SyntheticImage], 
        balance_metrics: Dict[str, Dict[str, int]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the generated dataset.
        
        Args:
            images: List of generated synthetic images
            balance_metrics: Balance metrics tracking
            
        Returns:
            Dictionary with dataset statistics
        """
        if not images:
            return {}
            
        # Extract all wave metrics
        heights = [img.ground_truth_labels.height_meters for img in images]
        directions = [img.ground_truth_labels.direction for img in images]
        breaking_types = [img.ground_truth_labels.breaking_type for img in images]
        height_confidences = [img.ground_truth_labels.height_confidence for img in images]
        direction_confidences = [img.ground_truth_labels.direction_confidence for img in images]
        breaking_confidences = [img.ground_truth_labels.breaking_confidence for img in images]
        
        statistics = {
            'total_images': len(images),
            'height_statistics': {
                'mean': float(np.mean(heights)),
                'std': float(np.std(heights)),
                'min': float(np.min(heights)),
                'max': float(np.max(heights)),
                'median': float(np.median(heights))
            },
            'confidence_statistics': {
                'height_confidence': {
                    'mean': float(np.mean(height_confidences)),
                    'std': float(np.std(height_confidences)),
                    'min': float(np.min(height_confidences))
                },
                'direction_confidence': {
                    'mean': float(np.mean(direction_confidences)),
                    'std': float(np.std(direction_confidences)),
                    'min': float(np.min(direction_confidences))
                },
                'breaking_confidence': {
                    'mean': float(np.mean(breaking_confidences)),
                    'std': float(np.std(breaking_confidences)),
                    'min': float(np.min(breaking_confidences))
                }
            },
            'distribution_balance': {
                'height_balance_score': self._calculate_balance_score(balance_metrics['height_distribution']),
                'direction_balance_score': self._calculate_balance_score(balance_metrics['direction_distribution']),
                'breaking_balance_score': self._calculate_balance_score(balance_metrics['breaking_type_distribution']),
                'weather_balance_score': self._calculate_balance_score(balance_metrics['weather_distribution'])
            },
            'extreme_conditions_count': sum(1 for img in images if img.ground_truth_labels.extreme_conditions),
            'generation_parameters': {
                'unique_seeds': len(set(img.generation_params.seed for img in images if img.generation_params.seed is not None)),
                'controlnet_scale_range': [
                    float(min(img.generation_params.controlnet_conditioning_scale for img in images)),
                    float(max(img.generation_params.controlnet_conditioning_scale for img in images))
                ]
            }
        }
        
        return statistics
    
    def _calculate_balance_score(self, distribution: Dict[str, int]) -> float:
        """Calculate balance score for a distribution (1.0 = perfectly balanced, 0.0 = completely unbalanced).
        
        Args:
            distribution: Dictionary with category counts
            
        Returns:
            Balance score between 0.0 and 1.0
        """
        if not distribution or sum(distribution.values()) == 0:
            return 0.0
            
        counts = list(distribution.values())
        total = sum(counts)
        
        if total == 0:
            return 0.0
            
        # Calculate expected count per category (perfect balance)
        expected_count = total / len(counts)
        
        # Calculate chi-square-like balance metric
        chi_square = sum((count - expected_count) ** 2 / expected_count for count in counts if expected_count > 0)
        
        # Normalize to [0, 1] where 1 is perfectly balanced
        # Use exponential decay to map chi-square to balance score
        balance_score = np.exp(-chi_square / total)
        
        return float(balance_score)
    
    def _validate_dataset_balance(
        self, 
        balance_metrics: Dict[str, Dict[str, int]], 
        target_size: int
    ) -> Dict[str, Any]:
        """Validate that the dataset is properly balanced across wave conditions.
        
        Args:
            balance_metrics: Balance metrics tracking
            target_size: Target dataset size
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_balanced': True,
            'balance_issues': [],
            'recommendations': []
        }
        
        # Check height distribution balance
        height_counts = list(balance_metrics['height_distribution'].values())
        if height_counts:
            height_cv = np.std(height_counts) / np.mean(height_counts) if np.mean(height_counts) > 0 else float('inf')
            if height_cv > 0.3:  # Coefficient of variation threshold
                validation_results['is_balanced'] = False
                validation_results['balance_issues'].append(f"Height distribution unbalanced (CV: {height_cv:.3f})")
                validation_results['recommendations'].append("Consider adjusting height range generation parameters")
        
        # Check direction distribution balance
        direction_counts = list(balance_metrics['direction_distribution'].values())
        if direction_counts:
            direction_cv = np.std(direction_counts) / np.mean(direction_counts) if np.mean(direction_counts) > 0 else float('inf')
            if direction_cv > 0.2:  # Stricter threshold for directions (only 3 categories)
                validation_results['is_balanced'] = False
                validation_results['balance_issues'].append(f"Direction distribution unbalanced (CV: {direction_cv:.3f})")
                validation_results['recommendations'].append("Ensure equal representation of LEFT/RIGHT/STRAIGHT directions")
        
        # Check breaking type distribution balance
        breaking_counts = list(balance_metrics['breaking_type_distribution'].values())
        if breaking_counts:
            breaking_cv = np.std(breaking_counts) / np.mean(breaking_counts) if np.mean(breaking_counts) > 0 else float('inf')
            if breaking_cv > 0.2:  # Stricter threshold for breaking types (only 3 categories)
                validation_results['is_balanced'] = False
                validation_results['balance_issues'].append(f"Breaking type distribution unbalanced (CV: {breaking_cv:.3f})")
                validation_results['recommendations'].append("Adjust weather conditions to achieve balanced breaking types")
        
        # Check minimum representation per category
        min_expected_per_category = target_size // 20  # At least 5% per major category
        
        for category_name, distribution in balance_metrics.items():
            for subcategory, count in distribution.items():
                if count < min_expected_per_category:
                    validation_results['is_balanced'] = False
                    validation_results['balance_issues'].append(
                        f"Low representation in {category_name}.{subcategory}: {count} < {min_expected_per_category}"
                    )
        
        # Add overall balance scores
        validation_results['balance_scores'] = {
            'height': self._calculate_balance_score(balance_metrics['height_distribution']),
            'direction': self._calculate_balance_score(balance_metrics['direction_distribution']),
            'breaking_type': self._calculate_balance_score(balance_metrics['breaking_type_distribution']),
            'weather': self._calculate_balance_score(balance_metrics['weather_distribution'])
        }
        
        return validation_results
    
    def validate_synthetic_vs_real_distribution(
        self,
        synthetic_dataset: LabeledDataset,
        real_data_statistics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate that synthetic data distribution matches real wave statistics.
        
        Args:
            synthetic_dataset: Generated synthetic dataset
            real_data_statistics: Optional real data statistics for comparison
            
        Returns:
            Dictionary with distribution validation results
        """
        validation_results = {
            'distribution_match': True,
            'comparison_metrics': {},
            'warnings': [],
            'recommendations': []
        }
        
        if real_data_statistics is None:
            # Use expected real-world wave statistics as baseline
            real_data_statistics = self._get_expected_real_wave_statistics()
            validation_results['warnings'].append("Using expected statistics - no real data provided for comparison")
        
        synthetic_stats = synthetic_dataset.statistics
        
        # Compare height distributions
        if 'height_statistics' in synthetic_stats and 'height_statistics' in real_data_statistics:
            height_comparison = self._compare_distributions(
                synthetic_stats['height_statistics'],
                real_data_statistics['height_statistics'],
                'height'
            )
            validation_results['comparison_metrics']['height'] = height_comparison
            
            if not height_comparison['is_similar']:
                validation_results['distribution_match'] = False
                validation_results['warnings'].append(f"Height distribution mismatch: {height_comparison['reason']}")
        
        # Compare confidence score distributions
        if 'confidence_statistics' in synthetic_stats and 'confidence_statistics' in real_data_statistics:
            for conf_type in ['height_confidence', 'direction_confidence', 'breaking_confidence']:
                if conf_type in synthetic_stats['confidence_statistics'] and conf_type in real_data_statistics['confidence_statistics']:
                    conf_comparison = self._compare_distributions(
                        synthetic_stats['confidence_statistics'][conf_type],
                        real_data_statistics['confidence_statistics'][conf_type],
                        conf_type
                    )
                    validation_results['comparison_metrics'][conf_type] = conf_comparison
                    
                    if not conf_comparison['is_similar']:
                        validation_results['distribution_match'] = False
                        validation_results['warnings'].append(f"{conf_type} distribution mismatch: {conf_comparison['reason']}")
        
        # Check extreme conditions ratio
        synthetic_extreme_ratio = synthetic_stats.get('extreme_conditions_count', 0) / synthetic_stats.get('total_images', 1)
        expected_extreme_ratio = real_data_statistics.get('extreme_conditions_ratio', 0.1)  # Expected ~10% extreme conditions
        
        if abs(synthetic_extreme_ratio - expected_extreme_ratio) > 0.05:  # 5% tolerance
            validation_results['distribution_match'] = False
            validation_results['warnings'].append(
                f"Extreme conditions ratio mismatch: synthetic={synthetic_extreme_ratio:.3f}, "
                f"expected={expected_extreme_ratio:.3f}"
            )
        
        # Add recommendations based on validation results
        if not validation_results['distribution_match']:
            validation_results['recommendations'].extend([
                "Consider adjusting generation parameters to better match real data",
                "Increase dataset size for better statistical representation",
                "Review weather condition distributions in generation process"
            ])
        
        # Calculate overall similarity score
        similarity_scores = [
            comp.get('similarity_score', 0.0) 
            for comp in validation_results['comparison_metrics'].values()
        ]
        validation_results['overall_similarity_score'] = np.mean(similarity_scores) if similarity_scores else 0.0
        
        self.logger.info(
            f"Synthetic vs real distribution validation: "
            f"match={validation_results['distribution_match']}, "
            f"similarity={validation_results['overall_similarity_score']:.3f}"
        )
        
        return validation_results
    
    def _get_expected_real_wave_statistics(self) -> Dict[str, Any]:
        """Get expected real-world wave statistics for comparison.
        
        Returns:
            Dictionary with expected real wave statistics
        """
        return {
            'height_statistics': {
                'mean': 2.5,      # Average wave height in meters
                'std': 1.2,       # Standard deviation
                'min': 0.5,       # Minimum observable height
                'max': 7.0,       # Maximum typical height
                'median': 2.2     # Median height
            },
            'confidence_statistics': {
                'height_confidence': {
                    'mean': 0.85,
                    'std': 0.12,
                    'min': 0.6
                },
                'direction_confidence': {
                    'mean': 0.82,
                    'std': 0.15,
                    'min': 0.5
                },
                'breaking_confidence': {
                    'mean': 0.80,
                    'std': 0.18,
                    'min': 0.4
                }
            },
            'extreme_conditions_ratio': 0.12,  # ~12% of waves are extreme
            'direction_distribution': {
                'LEFT': 0.35,
                'RIGHT': 0.35,
                'STRAIGHT': 0.30
            },
            'breaking_type_distribution': {
                'SPILLING': 0.45,
                'PLUNGING': 0.35,
                'SURGING': 0.20
            }
        }
    
    def _compare_distributions(
        self,
        synthetic_stats: Dict[str, float],
        real_stats: Dict[str, float],
        metric_name: str
    ) -> Dict[str, Any]:
        """Compare synthetic and real data distributions.
        
        Args:
            synthetic_stats: Synthetic data statistics
            real_stats: Real data statistics
            metric_name: Name of the metric being compared
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'is_similar': True,
            'similarity_score': 1.0,
            'reason': '',
            'differences': {}
        }
        
        # Compare mean values
        if 'mean' in synthetic_stats and 'mean' in real_stats:
            mean_diff = abs(synthetic_stats['mean'] - real_stats['mean'])
            mean_rel_diff = mean_diff / real_stats['mean'] if real_stats['mean'] != 0 else float('inf')
            comparison['differences']['mean_relative_diff'] = mean_rel_diff
            
            if mean_rel_diff > 0.2:  # 20% tolerance
                comparison['is_similar'] = False
                comparison['reason'] = f"Mean difference too large: {mean_rel_diff:.3f}"
        
        # Compare standard deviations
        if 'std' in synthetic_stats and 'std' in real_stats:
            std_diff = abs(synthetic_stats['std'] - real_stats['std'])
            std_rel_diff = std_diff / real_stats['std'] if real_stats['std'] != 0 else float('inf')
            comparison['differences']['std_relative_diff'] = std_rel_diff
            
            if std_rel_diff > 0.3:  # 30% tolerance for std
                comparison['is_similar'] = False
                if not comparison['reason']:
                    comparison['reason'] = f"Standard deviation difference too large: {std_rel_diff:.3f}"
        
        # Calculate overall similarity score
        mean_similarity = 1.0 - min(1.0, comparison['differences'].get('mean_relative_diff', 0.0))
        std_similarity = 1.0 - min(1.0, comparison['differences'].get('std_relative_diff', 0.0))
        comparison['similarity_score'] = (mean_similarity + std_similarity) / 2.0
        
        return comparison
    
    def cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            
        if self._controlnet is not None:
            del self._controlnet
            self._controlnet = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("FLUX ControlNet generator cleaned up")
        
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction