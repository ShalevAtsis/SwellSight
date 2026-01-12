"""
SwellSight Wave Analysis System

An AI-powered wave analysis system for surfers that transforms beach cam footage
into precise wave measurements including height, direction, and breaking type.
"""

__version__ = "0.1.0"
__author__ = "SwellSight Team"

from .core.pipeline import WaveAnalysisPipeline
from .core.depth_extractor import DepthExtractor
from .core.synthetic_generator import SyntheticDataGenerator
from .core.wave_analyzer import WaveAnalyzer

__all__ = [
    "WaveAnalysisPipeline",
    "DepthExtractor", 
    "SyntheticDataGenerator",
    "WaveAnalyzer"
]