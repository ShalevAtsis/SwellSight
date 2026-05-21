"""
Multi-task wave analysis model (training and inference).

Uses the same DINOv2 backbone and prediction heads as ``DINOv2WaveAnalyzer`` so
checkpoints trained with ``scripts/train.py`` load into the inference pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .backbone import DINOv2Backbone
from .heads import BreakingTypeHead, DirectionHead, WaveHeightHead

logger = logging.getLogger(__name__)

BACKBONE_NAME_MAP = {
    "dinov2-small": "dinov2_vits14",
    "dinov2-base": "dinov2_vitb14",
    "dinov2-large": "dinov2_vitl14",
    "dinov2-giant": "dinov2_vitg14",
    "facebook/dinov2-base": "dinov2_vitb14",
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
    "dinov2_vitg14": "dinov2_vitg14",
}


def resolve_backbone_name(name: str) -> str:
    return BACKBONE_NAME_MAP.get(name, name)


def _read_model_settings(config: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    if hasattr(config, "model"):
        model_conf = config.model
        return {
            "backbone_model": resolve_backbone_name(model_conf.backbone_model),
            "freeze_backbone": model_conf.freeze_backbone,
            "num_classes_direction": model_conf.num_classes_direction,
            "num_classes_breaking": model_conf.num_classes_breaking,
        }
    model_conf = config.get("model", {}) if isinstance(config, dict) else {}
    return {
        "backbone_model": resolve_backbone_name(
            model_conf.get("backbone", model_conf.get("backbone_model", "dinov2_vitb14"))
        ),
        "freeze_backbone": model_conf.get("freeze_backbone", True),
        "num_classes_direction": model_conf.get("num_classes_direction", 3),
        "num_classes_breaking": model_conf.get("num_classes_breaking", 3),
    }


class WaveAnalysisModel(nn.Module):
    """
    Multi-task wave model: height regression, direction and breaking classification.

    Checkpoint ``state_dict`` keys align with ``DINOv2WaveAnalyzer`` (``backbone``,
    ``height_head``, ``direction_head``, ``breaking_head``).
    """

    def __init__(self, config: Union[Dict[str, Any], Any]):
        super().__init__()
        settings = _read_model_settings(config)
        self.config = config
        self.backbone_model_name = settings["backbone_model"]

        self.backbone = DINOv2Backbone(
            model_name=settings["backbone_model"],
            freeze=settings["freeze_backbone"],
        )
        feature_dim = self.backbone.get_feature_dim()
        self.height_head = WaveHeightHead(feature_dim)
        self.direction_head = DirectionHead(
            feature_dim, num_classes=settings["num_classes_direction"]
        )
        self.breaking_head = BreakingTypeHead(
            feature_dim, num_classes=settings["num_classes_breaking"]
        )
        logger.info(
            "WaveAnalysisModel ready (%s, %s trainable params)",
            settings["backbone_model"],
            f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}",
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full head outputs (same keys as ``DINOv2WaveAnalyzer.forward``)."""
        features = self.backbone(x)
        return {
            **self.height_head(features),
            **self.direction_head(features),
            **self.breaking_head(features),
        }

    def forward_training(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Trainer-compatible output keys."""
        outputs = self.forward(x)
        return {
            "height": outputs["height_meters"].view(-1, 1),
            "direction": outputs["direction_logits"],
            "breaking_type": outputs["breaking_logits"],
        }


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Load a training checkpoint into a ``WaveAnalysisModel`` or ``DINOv2WaveAnalyzer``."""
    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.warning("Checkpoint missing keys: %s", missing[:5])
    if unexpected:
        logger.warning("Checkpoint unexpected keys: %s", unexpected[:5])
    return checkpoint
