"""
Training and optimization components for the SwellSight Wave Analysis System.

This module contains training logic, learning rate scheduling, and training callbacks
for the multi-task wave analysis model.
"""

from .trainer import WaveAnalysisTrainer
from .scheduler import create_lr_scheduler
from .callbacks import TrainingCallbacks

__all__ = [
    "WaveAnalysisTrainer",
    "create_lr_scheduler",
    "TrainingCallbacks"
]