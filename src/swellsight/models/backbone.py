"""
DINOv2 backbone integration for geometric understanding.

Provides frozen self-supervised features for multi-task wave analysis.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class BackboneModel(ABC):
    """Abstract base class for feature extraction backbones."""
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor."""
        pass

class DINOv2Backbone(BackboneModel, nn.Module):
    """DINOv2 backbone for geometric feature extraction with 4-channel input support."""
    
    def __init__(self, model_name: str = "dinov2_vitb14", freeze: bool = True):
        """Initialize DINOv2 backbone.
        
        Args:
            model_name: DINOv2 model variant (dinov2_vitb14, dinov2_vits14, dinov2_vitl14)
            freeze: Whether to freeze backbone weights
        """
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.feature_dim = self._get_feature_dim(model_name)
        self.input_channels = 4  # RGB + Depth
        self.target_size = (518, 518)  # DINOv2 input resolution
        
        # Initialize DINOv2 backbone
        self._backbone = self._load_dinov2_model(model_name)
        
        # Create 4-channel input adapter (RGB + Depth -> RGB)
        self.input_adapter = self._create_input_adapter()
        
        # Freeze backbone if requested
        if freeze:
            self._freeze_backbone()
            
        logger.info(f"Initialized DINOv2 backbone: {model_name}, frozen: {freeze}")
        
    def _get_feature_dim(self, model_name: str) -> int:
        """Get feature dimension for DINOv2 model variant."""
        feature_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536
        }
        return feature_dims.get(model_name, 768)
    
    def _load_dinov2_model(self, model_name: str) -> nn.Module:
        """Load DINOv2 model from torch hub."""
        try:
            # Load DINOv2 model from Facebook Research
            model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            logger.info(f"Successfully loaded {model_name} from torch hub")
            return model
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model {model_name}: {e}")
            # Fallback: create a dummy model for testing
            return self._create_dummy_backbone()
    
    def _create_dummy_backbone(self) -> nn.Module:
        """Create dummy backbone for testing when DINOv2 is unavailable."""
        logger.warning("Using dummy backbone - DINOv2 model unavailable")
        
        class DummyDINOv2(nn.Module):
            def __init__(self, feature_dim: int):
                super().__init__()
                self.feature_dim = feature_dim
                # Simple CNN to mimic DINOv2 feature extraction
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((16, 16)),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 8)),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, feature_dim)
                )
            
            def forward(self, x):
                return self.features(x)
        
        return DummyDINOv2(self.feature_dim)
    
    def _create_input_adapter(self) -> nn.Module:
        """Create adapter to process 4-channel input (RGB + Depth) for DINOv2."""
        return nn.Sequential(
            # Process 4-channel input
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output 3 channels for DINOv2
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()  # Normalize to [-1, 1] range
        )
    
    def _freeze_backbone(self):
        """Freeze DINOv2 backbone parameters."""
        for param in self._backbone.parameters():
            param.requires_grad = False
        logger.info("Frozen DINOv2 backbone parameters")
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess 4-channel input for DINOv2."""
        batch_size = x.shape[0]
        
        # Ensure input is 4-channel (RGB + Depth)
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4-channel input (RGB + Depth), got {x.shape[1]} channels")
        
        # Resize to target resolution if needed
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        # Normalize RGB channels to [0, 1]
        rgb = x[:, :3]  # RGB channels
        depth = x[:, 3:4]  # Depth channel
        
        # Normalize RGB to [0, 1] if not already
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # Normalize depth to [0, 1] if not already
        if depth.max() > 1.0:
            depth = depth / depth.max()
        
        # Recombine normalized channels
        x_normalized = torch.cat([rgb, depth], dim=1)
        
        return x_normalized
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 features from 4-channel input.
        
        Args:
            x: 4-channel tensor (RGB + Depth) of shape [B, 4, H, W]
            
        Returns:
            Feature tensor of shape [B, feature_dim]
        """
        # Preprocess 4-channel input
        x_processed = self._preprocess_input(x)
        
        # Convert 4-channel to 3-channel for DINOv2
        x_adapted = self.input_adapter(x_processed)
        
        # Normalize for DINOv2 (expects ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(x_adapted.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(x_adapted.device).view(1, 3, 1, 1)
        x_normalized = (x_adapted - mean) / std
        
        # Extract features using DINOv2
        with torch.set_grad_enabled(not self.freeze):
            features = self._backbone(x_normalized)
        
        return features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DINOv2 backbone.
        
        Args:
            x: 4-channel input tensor [B, 4, H, W]
            
        Returns:
            Feature tensor [B, feature_dim]
        """
        return self.extract_features(x)
    
    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.feature_dim