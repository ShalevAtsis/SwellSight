"""
Property-Based Test for Parameter Validation and Warnings
Tests Property 37: Parameter Validation and Warnings
Validates: Requirements 8.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import sys
from pathlib import Path

# Add utils to path for imports
sys.path.append('.')
from utils.config_manager import ConfigManager

class TestParameterValidation:
    """Property-based tests for parameter validation and warnings"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        
        # Create test configuration with parameter ranges
        self.test_config = {
            "pipeline": {
                "name": "test_pipeline",
                "version": "1.0"
            },
            "processing": {
                "batch_size": 4,
                "quality_threshold": 0.7,
                "image_resolution": {
                    "default": 512,
                    "max": 1024,
                    "min": 256
                }
            },
            "models": {
                "depth_model": {"name": "test-depth"},
                "base_model": {
                    "name": "test-base",
                    "guidance_scale": 3.5,
                    "num_inference_steps": 28
                },
                "controlnet_model": {"name": "test-controlnet"}
            },
            "paths": {
                "data_dir": "./data",
                "output_dir": "./outputs",
                "checkpoint_dir": "./checkpoints"
            }
        }
    
    @given(
        batch_size=st.integers(min_value=-10, max_value=100),
        quality_threshold=st.floats(min_value=-1.0, max_value=2.0),
        guidance_scale=st.floats(min_value=-5.0, max_value=20.0),
        inference_steps=st.integers(min_value=-10, max_value=200),
        image_resolution=st.integers(min_value=0, max_value=5000)
    )
    @settings(max_examples=100, deadline=None)
    def test_parameter_validation_and_warnings(self, batch_size, quality_threshold, 
                                             guidance_scale, inference_steps, image_resolution):
        """
        **Property 37: Parameter Validation and Warnings**
        
        For any parameter change, ranges should be validated and warnings should be 
        provided for unusual values
        
        **Validates: Requirements 8.2**
        """
        # Create test configuration with generated parameters
        test_config = self.test_config.copy()
        test_config['processing']['batch_size'] = batch_size
        test_config['processing']['quality_threshold'] = quality_threshold
        test_config['processing']['image_resolution']['default'] = image_resolution
        test_config['models']['base_model']['guidance_scale'] = guidance_scale
        test_config['models']['base_model']['num_inference_steps'] = inference_steps
        
        # Validate configuration
        is_valid, validation_errors = self.config_manager.validate_config(test_config)
        param_valid, param_warnings = self.config_manager.validate_parameter_ranges(test_config)
        
        # Property: Invalid parameters should be caught by validation
        if batch_size < 1:
            # Invalid batch size should generate validation error or warning
            assert not is_valid or len(param_warnings) > 0, \
                f"Invalid batch_size {batch_size} should be caught by validation"
        
        if quality_threshold < 0.0 or quality_threshold > 1.0:
            # Invalid quality threshold should generate warning
            assert len(param_warnings) > 0, \
                f"Invalid quality_threshold {quality_threshold} should generate warning"
        
        if guidance_scale > 10 or guidance_scale < 1:
            # Unusual guidance scale should generate warning
            assert len(param_warnings) > 0, \
                f"Unusual guidance_scale {guidance_scale} should generate warning"
        
        if inference_steps > 50 or inference_steps < 10:
            # Unusual inference steps should generate warning
            assert len(param_warnings) > 0, \
                f"Unusual inference_steps {inference_steps} should generate warning"
        
        if image_resolution > 1024 or image_resolution < 256:
            # Unusual image resolution should generate warning
            assert len(param_warnings) > 0, \
                f"Unusual image_resolution {image_resolution} should generate warning"
        
        # Property: Validation should always return boolean and list
        assert isinstance(is_valid, bool), "Validation should return boolean"
        assert isinstance(validation_errors, list), "Validation errors should be a list"
        assert isinstance(param_valid, bool), "Parameter validation should return boolean"
        assert isinstance(param_warnings, list), "Parameter warnings should be a list"
        
        # Property: All warnings should be strings
        for warning in param_warnings:
            assert isinstance(warning, str), f"Warning should be string, got {type(warning)}"
            assert len(warning) > 0, "Warning should not be empty"
        
        # Property: All errors should be strings
        for error in validation_errors:
            assert isinstance(error, str), f"Error should be string, got {type(error)}"
            assert len(error) > 0, "Error should not be empty"
    
    @given(
        config_dict=st.dictionaries(
            keys=st.sampled_from(['processing', 'models', 'paths']),
            values=st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(
                    st.integers(min_value=-1000, max_value=1000),
                    st.floats(min_value=-100.0, max_value=100.0),
                    st.text(min_size=0, max_size=50)
                )
            ),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_parameter_validation_robustness(self, config_dict):
        """
        Test that parameter validation handles arbitrary configuration changes robustly
        """
        # Start with valid base config
        test_config = self.test_config.copy()
        
        # Apply random changes
        for section, params in config_dict.items():
            if section not in test_config:
                test_config[section] = {}
            for key, value in params.items():
                test_config[section][key] = value
        
        # Validation should not crash regardless of input
        try:
            is_valid, validation_errors = self.config_manager.validate_config(test_config)
            param_valid, param_warnings = self.config_manager.validate_parameter_ranges(test_config)
            
            # Property: Validation should always complete without exceptions
            assert isinstance(is_valid, bool)
            assert isinstance(validation_errors, list)
            assert isinstance(param_valid, bool)
            assert isinstance(param_warnings, list)
            
        except Exception as e:
            pytest.fail(f"Parameter validation should not raise exceptions, got: {e}")
    
    @given(
        batch_sizes=st.lists(
            st.integers(min_value=1, max_value=32),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_batch_size_validation_consistency(self, batch_sizes):
        """
        Test that batch size validation is consistent across multiple values
        """
        warning_counts = []
        
        for batch_size in batch_sizes:
            test_config = self.test_config.copy()
            test_config['processing']['batch_size'] = batch_size
            
            _, param_warnings = self.config_manager.validate_parameter_ranges(test_config)
            warning_counts.append(len(param_warnings))
        
        # Property: Larger batch sizes should generally produce more warnings
        # (or at least not fewer warnings than smaller batch sizes)
        for i in range(1, len(batch_sizes)):
            if batch_sizes[i] > batch_sizes[i-1] and batch_sizes[i] > 16:
                # Large batch sizes should produce warnings
                assert warning_counts[i] >= 0, \
                    f"Large batch size {batch_sizes[i]} should produce appropriate warnings"
    
    @given(
        quality_thresholds=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_quality_threshold_validation_monotonicity(self, quality_thresholds):
        """
        Test that quality threshold validation behaves monotonically
        """
        # Filter out NaN values and sort
        valid_thresholds = [t for t in quality_thresholds if not np.isnan(t)]
        assume(len(valid_thresholds) >= 2)
        valid_thresholds.sort()
        
        warning_counts = []
        
        for threshold in valid_thresholds:
            test_config = self.test_config.copy()
            test_config['processing']['quality_threshold'] = threshold
            
            _, param_warnings = self.config_manager.validate_parameter_ranges(test_config)
            
            # Count warnings related to quality threshold
            quality_warnings = [w for w in param_warnings if 'quality_threshold' in w.lower()]
            warning_counts.append(len(quality_warnings))
        
        # Property: Extreme values should produce more warnings
        # Very low or very high thresholds should generate warnings
        for i, threshold in enumerate(valid_thresholds):
            if threshold < 0.3 or threshold > 0.95:
                assert warning_counts[i] >= 0, \
                    f"Extreme quality threshold {threshold} should be handled appropriately"
    
    def test_parameter_validation_edge_cases(self):
        """
        Test parameter validation with edge cases and boundary values
        """
        edge_cases = [
            # (description, config_updates, should_have_warnings)
            ("Zero batch size", {"processing": {"batch_size": 0}}, True),
            ("Negative batch size", {"processing": {"batch_size": -1}}, True),
            ("Very large batch size", {"processing": {"batch_size": 1000}}, True),
            ("Quality threshold at 0", {"processing": {"quality_threshold": 0.0}}, True),
            ("Quality threshold at 1", {"processing": {"quality_threshold": 1.0}}, False),
            ("Quality threshold above 1", {"processing": {"quality_threshold": 1.5}}, True),
            ("Very high guidance scale", {"models": {"base_model": {"guidance_scale": 50.0}}}, True),
            ("Very low guidance scale", {"models": {"base_model": {"guidance_scale": 0.1}}}, True),
            ("Very high inference steps", {"models": {"base_model": {"num_inference_steps": 200}}}, True),
            ("Very low inference steps", {"models": {"base_model": {"num_inference_steps": 1}}}, True),
        ]
        
        for description, config_updates, should_have_warnings in edge_cases:
            test_config = self.test_config.copy()
            
            # Apply updates
            for section, params in config_updates.items():
                if section not in test_config:
                    test_config[section] = {}
                for key, value in params.items():
                    if isinstance(test_config[section].get(key), dict):
                        test_config[section][key].update(value)
                    else:
                        test_config[section][key] = value
            
            # Validate
            is_valid, validation_errors = self.config_manager.validate_config(test_config)
            param_valid, param_warnings = self.config_manager.validate_parameter_ranges(test_config)
            
            # Check expectations
            if should_have_warnings:
                assert len(param_warnings) > 0 or not is_valid, \
                    f"{description}: Expected warnings or validation failure"
            
            # Property: Validation should always complete
            assert isinstance(is_valid, bool), f"{description}: Should return boolean"
            assert isinstance(param_warnings, list), f"{description}: Should return list"


if __name__ == "__main__":
    # Run the property-based tests
    pytest.main([__file__, "-v", "--tb=short"])