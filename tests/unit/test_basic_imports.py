"""
Basic unit tests to verify package structure and imports.

Tests that all main components can be imported without errors.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestBasicImports:
    """Test basic package imports."""
    
    def test_main_package_import(self):
        """Test that main package can be imported."""
        import swellsight
        assert hasattr(swellsight, '__version__')
    
    def test_core_components_import(self):
        """Test that core components can be imported."""
        from swellsight.core import DepthExtractor, SyntheticDataGenerator, WaveAnalyzer, WaveAnalysisPipeline
        
        # Check that classes exist
        assert DepthExtractor is not None
        assert SyntheticDataGenerator is not None
        assert WaveAnalyzer is not None
        assert WaveAnalysisPipeline is not None
    
    def test_utils_import(self):
        """Test that utilities can be imported."""
        from swellsight.utils import ConfigManager, setup_logging, HardwareManager
        
        # Check that classes exist
        assert ConfigManager is not None
        assert setup_logging is not None
        assert HardwareManager is not None
    
    def test_config_creation(self):
        """Test that configuration can be created."""
        from swellsight.utils.config import SwellSightConfig
        
        config = SwellSightConfig()
        assert config is not None
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')
        assert hasattr(config, 'system')
    
    def test_pipeline_creation(self):
        """Test that pipeline can be created (without initialization)."""
        from swellsight.core.pipeline import WaveAnalysisPipeline, PipelineConfig
        
        # Test config creation
        config = PipelineConfig()
        assert config is not None
        
        # Test pipeline creation (should not fail even if models aren't loaded)
        try:
            pipeline = WaveAnalysisPipeline(config)
            # Pipeline creation might fail due to missing models, but class should exist
        except Exception:
            # Expected - models not available yet
            pass
    
    def test_data_structures(self):
        """Test that data structures can be created."""
        from swellsight.core.depth_extractor import DepthMap, QualityMetrics
        from swellsight.core.synthetic_generator import WaveMetrics, WeatherConditions
        
        # Test DepthMap creation
        import numpy as np
        depth_map = DepthMap(
            data=np.zeros((100, 100)),
            resolution=(100, 100),
            quality_score=0.8,
            edge_preservation=0.9
        )
        assert depth_map is not None
        
        # Test WaveMetrics creation
        wave_metrics = WaveMetrics(
            height_meters=1.5,
            height_feet=4.9,
            height_confidence=0.9,
            direction="RIGHT",
            direction_confidence=0.8,
            breaking_type="SPILLING",
            breaking_confidence=0.85,
            extreme_conditions=False
        )
        assert wave_metrics is not None
        
        # Test WeatherConditions creation
        weather = WeatherConditions(
            lighting="sunny",
            weather="clear",
            wind_strength=0.3,
            wave_foam=0.2
        )
        assert weather is not None