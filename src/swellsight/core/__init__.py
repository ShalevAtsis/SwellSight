"""
Core pipeline components for the SwellSight Wave Analysis System.

This module contains the three main stages of the wave analysis pipeline:
- Stage A: Depth extraction using Depth-Anything-V2
- Stage B: Synthetic data generation using FLUX ControlNet
- Stage C: Multi-task wave analysis using DINOv2 backbone
"""

from .depth_extractor import DepthExtractor
from .synthetic_generator import SyntheticDataGenerator
from .wave_analyzer import WaveAnalyzer
from .pipeline import WaveAnalysisPipeline

__all__ = [
    "DepthExtractor",
    "SyntheticDataGenerator", 
    "WaveAnalyzer",
    "WaveAnalysisPipeline"
]