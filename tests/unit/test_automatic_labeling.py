"""
Unit tests for the automatic labeling system functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from swellsight.core.synthetic_generator import (
    AutomaticLabelingSystem,
    WeatherConditions,
    WaveCharacteristics,
    WaveMetrics,
    LabelValidationResult
)
from swellsight.core.depth_extractor import DepthMap


class TestAutomaticLabelingSystem:
    """Test the AutomaticLabelingSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.labeling_system = AutomaticLabelingSystem()
        
        # Create a test depth map with wave-like patterns
        self.depth_data = np.zeros((256, 256), dtype=np.float32)
        
        # Add some wave-like patterns
        x = np.linspace(0, 4 * np.pi, 256)
        y = np.linspace(0, 2 * np.pi, 256)
        X, Y = np.meshgrid(x, y)
        
        # Create wave patterns
        self.depth_data = 0.5 + 0.3 * np.sin(X) * np.cos(Y/2) + 0.1 * np.random.rand(256, 256)
        self.depth_data = np.clip(self.depth_data, 0, 1)
        
        self.depth_map = DepthMap(
            data=self.depth_data,
            resolution=(256, 256),
            quality_score=0.8,
            edge_preservation=0.7
        )
        
        self.conditions = WeatherConditions(
            lighting="sunny",
            weather="clear",
            wind_strength=0.5,
            wave_foam=0.4
        )
    
    def test_extract_wave_characteristics(self):
        """Test wave characteristics extraction."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        
        assert isinstance(characteristics, WaveCharacteristics)
        assert isinstance(characteristics.peak_positions, list)
        assert isinstance(characteristics.wave_count, int)
        assert characteristics.wave_count >= 0
        assert characteristics.dominant_wavelength > 0
        assert characteristics.wave_steepness >= 0
        assert characteristics.breaking_intensity >= 0
        assert 0 <= characteristics.foam_coverage <= 1
        assert characteristics.depth_gradient_magnitude >= 0
        assert characteristics.surface_roughness >= 0
    
    def test_estimate_wave_height(self):
        """Test wave height estimation."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        height_meters, confidence = self.labeling_system.estimate_wave_height(
            characteristics, self.depth_map
        )
        
        assert 0.5 <= height_meters <= 8.0
        assert 0.0 <= confidence <= 1.0
        assert isinstance(height_meters, float)
        assert isinstance(confidence, float)
    
    def test_classify_wave_direction(self):
        """Test wave direction classification."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        direction, confidence = self.labeling_system.classify_wave_direction(
            characteristics, self.depth_map
        )
        
        assert direction in ["LEFT", "RIGHT", "STRAIGHT"]
        assert 0.0 <= confidence <= 1.0
        assert isinstance(direction, str)
        assert isinstance(confidence, float)
    
    def test_classify_breaking_type(self):
        """Test breaking type classification."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        breaking_type, confidence = self.labeling_system.classify_breaking_type(
            characteristics, self.conditions
        )
        
        assert breaking_type in ["SPILLING", "PLUNGING", "SURGING"]
        assert 0.0 <= confidence <= 1.0
        assert isinstance(breaking_type, str)
        assert isinstance(confidence, float)
    
    def test_classify_breaking_type_wind_conditions(self):
        """Test breaking type classification with different wind conditions."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        
        # High wind should favor spilling
        high_wind_conditions = WeatherConditions("stormy", "clear", 0.8, 0.7)
        breaking_type, confidence = self.labeling_system.classify_breaking_type(
            characteristics, high_wind_conditions
        )
        
        assert breaking_type == "SPILLING"
        assert confidence > 0.8
    
    def test_validate_labels(self):
        """Test label validation."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        
        # Create valid wave metrics
        wave_metrics = WaveMetrics(
            height_meters=2.5,
            height_feet=2.5 * 3.28084,
            height_confidence=0.9,
            direction="RIGHT",
            direction_confidence=0.85,
            breaking_type="PLUNGING",
            breaking_confidence=0.8,
            extreme_conditions=False
        )
        
        validation_result = self.labeling_system.validate_labels(
            wave_metrics, characteristics, self.conditions
        )
        
        assert isinstance(validation_result, LabelValidationResult)
        assert isinstance(validation_result.is_valid, bool)
        assert 0.0 <= validation_result.confidence_score <= 1.0
        assert isinstance(validation_result.validation_errors, list)
        assert isinstance(validation_result.consistency_warnings, list)
        assert isinstance(validation_result.quality_metrics, dict)
    
    def test_validate_labels_invalid_height(self):
        """Test label validation with invalid height."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        
        # Create invalid wave metrics (height too high)
        wave_metrics = WaveMetrics(
            height_meters=10.0,  # Invalid - too high
            height_feet=10.0 * 3.28084,
            height_confidence=0.9,
            direction="RIGHT",
            direction_confidence=0.85,
            breaking_type="PLUNGING",
            breaking_confidence=0.8,
            extreme_conditions=True
        )
        
        validation_result = self.labeling_system.validate_labels(
            wave_metrics, characteristics, self.conditions
        )
        
        assert not validation_result.is_valid
        assert len(validation_result.validation_errors) > 0
        assert any("height" in error.lower() for error in validation_result.validation_errors)
    
    def test_validate_labels_invalid_direction(self):
        """Test label validation with invalid direction."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        
        # Create invalid wave metrics (invalid direction)
        wave_metrics = WaveMetrics(
            height_meters=2.5,
            height_feet=2.5 * 3.28084,
            height_confidence=0.9,
            direction="INVALID",  # Invalid direction
            direction_confidence=0.85,
            breaking_type="PLUNGING",
            breaking_confidence=0.8,
            extreme_conditions=False
        )
        
        validation_result = self.labeling_system.validate_labels(
            wave_metrics, characteristics, self.conditions
        )
        
        assert not validation_result.is_valid
        assert len(validation_result.validation_errors) > 0
        assert any("direction" in error.lower() for error in validation_result.validation_errors)
    
    def test_validate_labels_consistency_warnings(self):
        """Test label validation with consistency warnings."""
        characteristics = self.labeling_system.extract_wave_characteristics(self.depth_map)
        
        # Create conditions that should trigger warnings
        high_wind_conditions = WeatherConditions("stormy", "clear", 0.8, 0.7)
        
        # But use non-spilling breaking type
        wave_metrics = WaveMetrics(
            height_meters=2.5,
            height_feet=2.5 * 3.28084,
            height_confidence=0.9,
            direction="RIGHT",
            direction_confidence=0.85,
            breaking_type="SURGING",  # Inconsistent with high wind
            breaking_confidence=0.8,
            extreme_conditions=False
        )
        
        validation_result = self.labeling_system.validate_labels(
            wave_metrics, characteristics, high_wind_conditions
        )
        
        # Should be valid but have warnings
        assert validation_result.is_valid
        assert len(validation_result.consistency_warnings) > 0
    
    def test_edge_case_flat_depth_map(self):
        """Test with flat depth map (no waves)."""
        flat_depth_data = np.full((256, 256), 0.5, dtype=np.float32)
        flat_depth_map = DepthMap(
            data=flat_depth_data,
            resolution=(256, 256),
            quality_score=0.8,
            edge_preservation=0.7
        )
        
        characteristics = self.labeling_system.extract_wave_characteristics(flat_depth_map)
        
        # Should handle flat case gracefully
        assert characteristics.wave_count >= 0
        assert characteristics.dominant_wavelength > 0
        assert characteristics.wave_steepness >= 0
        
        height_meters, confidence = self.labeling_system.estimate_wave_height(
            characteristics, flat_depth_map
        )
        
        # Should return minimum height for flat case
        assert height_meters >= 0.5
        assert confidence > 0
    
    def test_edge_case_noisy_depth_map(self):
        """Test with very noisy depth map."""
        noisy_depth_data = np.random.rand(256, 256).astype(np.float32)
        noisy_depth_map = DepthMap(
            data=noisy_depth_data,
            resolution=(256, 256),
            quality_score=0.3,
            edge_preservation=0.2
        )
        
        characteristics = self.labeling_system.extract_wave_characteristics(noisy_depth_map)
        
        # Should handle noisy case gracefully
        assert isinstance(characteristics.wave_count, int)
        assert characteristics.wave_count >= 0
        assert characteristics.dominant_wavelength > 0
        
        height_meters, confidence = self.labeling_system.estimate_wave_height(
            characteristics, noisy_depth_map
        )
        
        # Should return valid height even for noisy input
        assert 0.5 <= height_meters <= 8.0
        assert 0.0 <= confidence <= 1.0


class TestAutomaticLabelingIntegration:
    """Integration tests for automatic labeling with synthetic generator."""
    
    def test_labeling_system_initialization(self):
        """Test that labeling system is properly initialized in generator."""
        from swellsight.core.synthetic_generator import FLUXControlNetGenerator
        
        generator = FLUXControlNetGenerator()
        
        assert hasattr(generator, 'labeling_system')
        assert isinstance(generator.labeling_system, AutomaticLabelingSystem)
        assert generator.labeling_system.logger is not None
    
    def test_extract_wave_metrics_integration(self):
        """Test wave metrics extraction integration."""
        from swellsight.core.synthetic_generator import FLUXControlNetGenerator, GenerationConfig
        
        generator = FLUXControlNetGenerator()
        
        # Create test depth map
        depth_data = np.random.rand(256, 256).astype(np.float32)
        depth_map = DepthMap(
            data=depth_data,
            resolution=(256, 256),
            quality_score=0.8,
            edge_preservation=0.7
        )
        
        conditions = WeatherConditions("sunny", "clear", 0.5, 0.4)
        config = GenerationConfig()
        
        # Test the extraction method
        metrics = generator._extract_wave_metrics(depth_map, conditions, config)
        
        assert isinstance(metrics, WaveMetrics)
        assert 0.5 <= metrics.height_meters <= 8.0
        assert metrics.height_feet == metrics.height_meters * 3.28084
        assert metrics.direction in ["LEFT", "RIGHT", "STRAIGHT"]
        assert metrics.breaking_type in ["SPILLING", "PLUNGING", "SURGING"]
        assert 0.0 <= metrics.height_confidence <= 1.0
        assert 0.0 <= metrics.direction_confidence <= 1.0
        assert 0.0 <= metrics.breaking_confidence <= 1.0
        assert isinstance(metrics.extreme_conditions, bool)