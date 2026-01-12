"""
Data processing and management for the SwellSight Wave Analysis System.

This module handles image preprocessing, data augmentation, dataset creation,
and data loading utilities for training and inference.
"""

from .preprocessing import BeachCamImage, ImageProcessor
from .augmentation import WaveAugmentation
from .datasets import WaveDataset, SyntheticWaveDataset
from .loaders import create_data_loaders

__all__ = [
    "BeachCamImage",
    "ImageProcessor", 
    "WaveAugmentation",
    "WaveDataset",
    "SyntheticWaveDataset",
    "create_data_loaders"
]