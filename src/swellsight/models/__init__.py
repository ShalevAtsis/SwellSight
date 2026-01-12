"""
Model architectures and components for the SwellSight Wave Analysis System.

This module contains neural network architectures, loss functions, and model utilities
for the multi-task wave analysis system.
"""

from .backbone import DINOv2Backbone
from .heads import WaveHeightHead, DirectionHead, BreakingTypeHead
from .losses import MultiTaskLoss

__all__ = [
    "DINOv2Backbone",
    "WaveHeightHead", 
    "DirectionHead",
    "BreakingTypeHead",
    "MultiTaskLoss"
]