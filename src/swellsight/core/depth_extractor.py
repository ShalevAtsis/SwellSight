"""
Stage A: Depth Extraction Engine

Converts 2D beach cam images into high-sensitivity depth maps using Depth-Anything-V2.
Preserves sharp wave edges and captures fine-grained water surface texture.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

@dataclass
class DepthMap:
    """Normalized depth map with quality metrics."""
    data: np.ndarray  # Normalized depth values [0,1]
    resolution: Tuple[int, int]
    quality_score: float
    edge_preservation: float

@dataclass 
class QualityMetrics:
    """Statistical quality assessment metrics for depth maps."""
    overall_score: float
    edge_preservation: float
    texture_capture: float
    far_field_sensitivity: float
    contrast_ratio: float

class DepthExtractor(ABC):
    """Abstract base class for depth extraction engines."""
    
    @abstractmethod
    def extract_depth(self, image: np.ndarray) -> DepthMap:
        """Extract normalized depth map from beach cam image.
        
        Args:
            image: RGB beach cam image as numpy array
            
        Returns:
            DepthMap with normalized depth values and quality metrics
        """
        pass
    
    @abstractmethod
    def validate_quality(self, depth_map: DepthMap) -> QualityMetrics:
        """Assess depth map quality using statistical measures.
        
        Args:
            depth_map: Generated depth map to validate
            
        Returns:
            QualityMetrics with detailed quality assessment
        """
        pass

class DepthAnythingV2Extractor(DepthExtractor):
    """Depth-Anything-V2 implementation for marine environment depth extraction."""
    
    def __init__(self, model_size: str = "large", precision: str = "fp16", device: Optional[str] = None):
        """Initialize Depth-Anything-V2 extractor.
        
        Args:
            model_size: Model size variant ("small", "base", "large")
            precision: Precision mode ("fp16", "fp32")
            device: Device to run model on ("cuda", "cpu", or None for auto-detect)
        """
        self.model_size = model_size
        self.precision = precision
        self.input_resolution = (518, 518)
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Model configuration
        model_configs = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf", 
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        if model_size not in model_configs:
            raise ValueError(f"Invalid model_size: {model_size}. Must be one of {list(model_configs.keys())}")
            
        self.model_name = model_configs[model_size]
        self._model = None
        self._processor = None
        
        logger.info(f"Initializing Depth-Anything-V2 {model_size} on {self.device} with {precision} precision")
        
    def _load_model(self):
        """Lazy load the model and processor."""
        if self._model is None:
            try:
                # Load processor and model
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                self._model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
                
                # Move to device and set precision
                self._model = self._model.to(self.device)
                
                if self.precision == "fp16" and self.device == "cuda":
                    self._model = self._model.half()
                    
                self._model.eval()
                logger.info(f"Successfully loaded {self.model_name} on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[Image.Image, Tuple[int, int]]:
        """Preprocess image for depth extraction.
        
        Args:
            image: Input RGB image as numpy array (H, W, 3)
            
        Returns:
            Tuple of (preprocessed PIL image, original size)
        """
        # Validate input
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
            
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Convert to PIL Image
        if image.dtype != np.uint8:
            # Normalize to 0-255 if not already uint8
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
                
        pil_image = Image.fromarray(image)
        
        return pil_image, original_size
    
    def _postprocess_depth(self, depth_tensor: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess depth tensor to numpy array.
        
        Args:
            depth_tensor: Raw depth prediction tensor
            target_size: Target size (width, height) for output
            
        Returns:
            Normalized depth map as numpy array
        """
        # Convert to numpy
        if depth_tensor.is_cuda:
            depth_np = depth_tensor.cpu().numpy()
        else:
            depth_np = depth_tensor.numpy()
            
        # Remove batch dimension if present
        if len(depth_np.shape) == 4:
            depth_np = depth_np.squeeze(0)
        if len(depth_np.shape) == 3:
            depth_np = depth_np.squeeze(0)
            
        # Resize to target size
        depth_resized = cv2.resize(depth_np, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] range
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        if depth_max > depth_min:
            depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_resized)
            
        return depth_normalized
    
    def _calculate_quality_metrics(self, depth_map: np.ndarray, original_image: np.ndarray) -> Tuple[float, float]:
        """Calculate basic quality metrics for the depth map.
        
        Args:
            depth_map: Normalized depth map
            original_image: Original RGB image
            
        Returns:
            Tuple of (quality_score, edge_preservation)
        """
        # Convert original image to grayscale for edge detection
        if len(original_image.shape) == 3:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = original_image
            
        # Resize gray image to match depth map
        gray_resized = cv2.resize(gray_image, (depth_map.shape[1], depth_map.shape[0]))
        
        # Calculate edge preservation using Sobel operators
        # Original image edges
        sobel_x_orig = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_orig = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        edges_orig = np.sqrt(sobel_x_orig**2 + sobel_y_orig**2)
        
        # Depth map edges
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        sobel_x_depth = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_depth = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edges_depth = np.sqrt(sobel_x_depth**2 + sobel_y_depth**2)
        
        # Normalize edges
        edges_orig_norm = edges_orig / (edges_orig.max() + 1e-8)
        edges_depth_norm = edges_depth / (edges_depth.max() + 1e-8)
        
        # Calculate correlation between edge maps
        edge_preservation = np.corrcoef(edges_orig_norm.flatten(), edges_depth_norm.flatten())[0, 1]
        edge_preservation = max(0.0, edge_preservation)  # Ensure non-negative
        
        # Calculate overall quality score based on depth map statistics
        depth_std = np.std(depth_map)
        depth_range = np.ptp(depth_map)  # Peak-to-peak (max - min)
        
        # Quality score combines variation and range
        quality_score = min(1.0, (depth_std * 2 + depth_range) / 2)
        
        return quality_score, edge_preservation
        
    def extract_depth(self, image: np.ndarray) -> DepthMap:
        """Extract depth map using Depth-Anything-V2.
        
        Args:
            image: RGB beach cam image as numpy array (H, W, 3)
            
        Returns:
            DepthMap with normalized depth values and quality metrics
        """
        self._load_model()
        
        try:
            # Preprocess image
            pil_image, original_size = self._preprocess_image(image)
            
            # Prepare inputs using the processor
            inputs = self._processor(images=pil_image, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Apply precision if using fp16
            if self.precision == "fp16" and self.device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                
            # Extract depth prediction
            predicted_depth = outputs.predicted_depth
            
            # Postprocess to get normalized depth map
            depth_array = self._postprocess_depth(predicted_depth, original_size)
            
            # Calculate quality metrics
            quality_score, edge_preservation = self._calculate_quality_metrics(depth_array, image)
            
            # Create DepthMap object
            depth_map = DepthMap(
                data=depth_array,
                resolution=original_size,
                quality_score=quality_score,
                edge_preservation=edge_preservation
            )
            
            logger.info(f"Extracted depth map: size={original_size}, quality={quality_score:.3f}, edge_preservation={edge_preservation:.3f}")
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth extraction failed: {e}")
            raise RuntimeError(f"Depth extraction failed: {e}")
        
    def validate_quality(self, depth_map: DepthMap) -> QualityMetrics:
        """Validate depth map quality for marine environments.
        
        Args:
            depth_map: Generated depth map to validate
            
        Returns:
            QualityMetrics with detailed quality assessment
        """
        try:
            # Extract depth data
            depth_data = depth_map.data
            
            # 1. Overall quality score (already calculated in depth_map)
            overall_score = depth_map.quality_score
            
            # 2. Edge preservation (already calculated in depth_map)
            edge_preservation = depth_map.edge_preservation
            
            # 3. Texture capture assessment
            texture_capture = self._assess_texture_capture(depth_data)
            
            # 4. Far-field depth sensitivity
            far_field_sensitivity = self._assess_far_field_sensitivity(depth_data)
            
            # 5. Wave-ocean contrast ratio
            contrast_ratio = self._calculate_contrast_ratio(depth_data)
            
            # Create comprehensive quality metrics
            quality_metrics = QualityMetrics(
                overall_score=overall_score,
                edge_preservation=edge_preservation,
                texture_capture=texture_capture,
                far_field_sensitivity=far_field_sensitivity,
                contrast_ratio=contrast_ratio
            )
            
            logger.info(f"Quality validation complete: overall={overall_score:.3f}, "
                       f"edge={edge_preservation:.3f}, texture={texture_capture:.3f}, "
                       f"far_field={far_field_sensitivity:.3f}, contrast={contrast_ratio:.3f}")
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            # Return default metrics on failure
            return QualityMetrics(
                overall_score=0.0,
                edge_preservation=0.0,
                texture_capture=0.0,
                far_field_sensitivity=0.0,
                contrast_ratio=0.0
            )
    
    def _assess_texture_capture(self, depth_data: np.ndarray) -> float:
        """Assess how well the depth map captures fine-grained texture.
        
        Args:
            depth_data: Normalized depth map data
            
        Returns:
            Texture capture score [0, 1]
        """
        # Calculate local variation using different kernel sizes
        kernel_sizes = [3, 5, 7]
        texture_scores = []
        
        for kernel_size in kernel_sizes:
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            local_mean = cv2.filter2D(depth_data, -1, kernel)
            local_variation = np.abs(depth_data - local_mean)
            avg_variation = np.mean(local_variation)
            texture_scores.append(avg_variation)
        
        # Combine scores from different scales
        texture_score = np.mean(texture_scores)
        
        # Normalize to [0, 1] range (empirically determined scaling)
        texture_score = min(1.0, texture_score * 10)
        
        return texture_score
    
    def _assess_far_field_sensitivity(self, depth_data: np.ndarray) -> float:
        """Assess depth sensitivity for distant objects.
        
        Args:
            depth_data: Normalized depth map data
            
        Returns:
            Far-field sensitivity score [0, 1]
        """
        height, width = depth_data.shape
        
        # Divide image into near-field (bottom 1/3) and far-field (top 1/3)
        near_field = depth_data[2*height//3:, :]
        far_field = depth_data[:height//3, :]
        
        # Calculate variation in each field
        near_variation = np.std(near_field)
        far_variation = np.std(far_field)
        
        # Far-field sensitivity is the ratio of far-field to near-field variation
        # Good depth maps should maintain reasonable variation in far-field
        if near_variation > 0:
            sensitivity_ratio = far_variation / near_variation
        else:
            sensitivity_ratio = far_variation
        
        # Normalize to [0, 1] range
        sensitivity_score = min(1.0, sensitivity_ratio * 2)
        
        return sensitivity_score
    
    def _calculate_contrast_ratio(self, depth_data: np.ndarray) -> float:
        """Calculate wave-ocean contrast ratio for enhanced visibility.
        
        Args:
            depth_data: Normalized depth map data
            
        Returns:
            Contrast ratio [0, 1]
        """
        # Calculate histogram to find dominant depth values (ocean surface)
        hist, bin_edges = np.histogram(depth_data.flatten(), bins=50, range=(0, 1))
        
        # Find the most common depth value (likely ocean surface)
        dominant_bin_idx = np.argmax(hist)
        ocean_depth = (bin_edges[dominant_bin_idx] + bin_edges[dominant_bin_idx + 1]) / 2
        
        # Calculate contrast as the standard deviation of depths relative to ocean surface
        depth_deviations = np.abs(depth_data - ocean_depth)
        contrast_ratio = np.std(depth_deviations)
        
        # Normalize to [0, 1] range
        contrast_ratio = min(1.0, contrast_ratio * 4)
        
        return contrast_ratio
    
    def normalize_depth_for_waves(self, depth_map: DepthMap, enhancement_factor: float = 2.0) -> DepthMap:
        """Normalize depth map to enhance wave-ocean contrast.
        
        Args:
            depth_map: Input depth map
            enhancement_factor: Factor to enhance wave visibility
            
        Returns:
            Enhanced depth map with better wave-ocean contrast
        """
        try:
            depth_data = depth_map.data.copy()
            
            # 1. Histogram equalization for better contrast
            depth_uint8 = (depth_data * 255).astype(np.uint8)
            equalized = cv2.equalizeHist(depth_uint8)
            depth_equalized = equalized.astype(np.float32) / 255.0
            
            # 2. Enhance wave features using morphological operations
            # Create kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # Apply opening to remove noise
            depth_uint8_eq = (depth_equalized * 255).astype(np.uint8)
            opened = cv2.morphologyEx(depth_uint8_eq, cv2.MORPH_OPEN, kernel)
            depth_opened = opened.astype(np.float32) / 255.0
            
            # 3. Apply Gaussian blur to smooth while preserving edges
            depth_smoothed = cv2.GaussianBlur(depth_opened, (3, 3), 0.5)
            
            # 4. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            depth_uint8_smooth = (depth_smoothed * 255).astype(np.uint8)
            enhanced = clahe.apply(depth_uint8_smooth)
            depth_enhanced = enhanced.astype(np.float32) / 255.0
            
            # 5. Apply enhancement factor
            # Find median depth (likely ocean surface)
            median_depth = np.median(depth_enhanced)
            
            # Enhance deviations from median
            deviations = depth_enhanced - median_depth
            enhanced_deviations = deviations * enhancement_factor
            
            # Combine back and normalize
            depth_final = median_depth + enhanced_deviations
            depth_final = np.clip(depth_final, 0, 1)
            
            # Recalculate quality metrics for enhanced depth map
            quality_score, edge_preservation = self._calculate_quality_metrics(
                depth_final, 
                np.random.randint(0, 255, (*depth_final.shape, 3), dtype=np.uint8)  # Dummy image for metrics
            )
            
            # Create enhanced depth map
            enhanced_depth_map = DepthMap(
                data=depth_final,
                resolution=depth_map.resolution,
                quality_score=quality_score,
                edge_preservation=edge_preservation
            )
            
            logger.info(f"Depth normalization complete: enhancement_factor={enhancement_factor}, "
                       f"new_quality={quality_score:.3f}")
            
            return enhanced_depth_map
            
        except Exception as e:
            logger.error(f"Depth normalization failed: {e}")
            return depth_map  # Return original on failure