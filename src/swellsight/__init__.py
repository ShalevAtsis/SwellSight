"""
SwellSight Wave Analysis System

Heavy ML imports are lazy so lightweight entrypoints (Alembic, API platform mode)
do not require torch/psutil at import time.
"""

__version__ = "0.1.0"
__author__ = "SwellSight Team"

__all__ = [
    "WaveAnalysisPipeline",
    "DepthExtractor",
    "SyntheticDataGenerator",
    "WaveAnalyzer",
]


def __getattr__(name: str):
    if name == "WaveAnalysisPipeline":
        from .core.pipeline import WaveAnalysisPipeline

        return WaveAnalysisPipeline
    if name == "DepthExtractor":
        from .core.depth_extractor import DepthExtractor

        return DepthExtractor
    if name == "SyntheticDataGenerator":
        from .core.synthetic_generator import SyntheticDataGenerator

        return SyntheticDataGenerator
    if name == "WaveAnalyzer":
        from .core.wave_analyzer import WaveAnalyzer

        return WaveAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")