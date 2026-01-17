import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Union
from pathlib import Path

class WaveAnalysisModel(nn.Module):
    """
    Multi-task Wave Analysis Model using DINOv2 Backbone.
    Predicts: Wave Height (Regression), Direction (Classification), Breaking Type (Classification)
    """
    def __init__(self, config: Union[Dict[str, Any], Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Handle both dict and config object
        if hasattr(config, 'model'):
            # Config object (SwellSightConfig)
            model_conf = config.model
            backbone_name = model_conf.backbone_model
            freeze_backbone = model_conf.freeze_backbone
            input_channels = 4  # RGB + Depth
            num_classes_direction = model_conf.num_classes_direction
            num_classes_breaking = model_conf.num_classes_breaking
        else:
            # Dict config
            model_conf = config.get('model', {})
            backbone_name = model_conf.get('backbone', 'dinov2_vitb14')
            freeze_backbone = model_conf.get('freeze_backbone', True)
            input_channels = model_conf.get('input_channels', 4)
            num_classes_direction = model_conf.get('num_classes_direction', 3)
            num_classes_breaking = model_conf.get('num_classes_breaking', 3)
        
        # Map config names to torch.hub names
        backbone_mapping = {
            'dinov2-small': 'dinov2_vits14',
            'dinov2-base': 'dinov2_vitb14',
            'dinov2-large': 'dinov2_vitl14',
            'dinov2-giant': 'dinov2_vitg14',
        }
        backbone_name = backbone_mapping.get(backbone_name, backbone_name)
        
        self.config = config
        
        # 1. Load Backbone (DINOv2)
        self.logger.info(f"Loading backbone: {backbone_name}...")
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name, trust_repo=True)
        except Exception as e:
            self.logger.warning(f"Failed to load from torch.hub: {e}. Using fallback initialization.")
            # Fallback: create a simple backbone for testing
            self.backbone = self._create_fallback_backbone()
        
        # DINOv2 embedding dimensions
        if 'vits' in backbone_name: 
            embed_dim = 384
        elif 'vitb' in backbone_name: 
            embed_dim = 768
        elif 'vitl' in backbone_name: 
            embed_dim = 1024
        else: 
            embed_dim = 768  # Default fallback
        
        # 2. Input Adapter (Critical for RGB+Depth)
        # DINOv2 expects 3 channels. We have 4 (RGB + Depth).
        # We use a 1x1 convolution to learn the optimal mapping from 4->3 channels.
        self.input_channels = input_channels
        if self.input_channels != 3:
            self.input_adapter = nn.Conv2d(self.input_channels, 3, kernel_size=1)
            self.logger.info(f"Created input adapter: {self.input_channels} -> 3 channels")
        else:
            self.input_adapter = nn.Identity()

        # 3. Freeze Backbone (Optional)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.logger.info("[OK] Backbone frozen")
        else:
            self.logger.info("[OK] Backbone trainable")

        # 4. Task Heads
        # Height: Regression (1 value)
        self.height_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) 
        )
        
        # Direction: Classification (Left, Right, Straight)
        self.direction_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_direction) 
        )
        
        # Breaking Type: Classification (Spilling, Plunging, Surging)
        self.breaking_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_breaking)
        )
        
        self.logger.info("[OK] Model initialized successfully")
    
    def _create_fallback_backbone(self):
        """Create a simple fallback backbone for testing when torch.hub fails."""
        class FallbackBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                self.embed_dim = 768
                self.projection = nn.Linear(64, 768)
            
            def forward_features(self, x):
                features = self.features(x)
                cls_token = self.projection(features)
                return {'x_norm_clstoken': cls_token}
        
        return FallbackBackbone()

    def forward(self, x):
        # x shape: (Batch, 4, H, W) or (Batch, 3, H, W)
        
        # 1. Adapt Input (4->3 channels if needed)
        x = self.input_adapter(x)
        
        # 2. Backbone Features
        # DINOv2 forward_features returns a dict. 'x_norm_clstoken' is the CLS token output.
        features_dict = self.backbone.forward_features(x)
        cls_token = features_dict['x_norm_clstoken'] # (Batch, embed_dim)
        
        # 3. Task Predictions
        height = self.height_head(cls_token) # (Batch, 1)
        direction = self.direction_head(cls_token) # (Batch, 3)
        breaking = self.breaking_head(cls_token) # (Batch, 3)
        
        return {
            "height": height,
            "direction": direction,
            "breaking_type": breaking
        }
