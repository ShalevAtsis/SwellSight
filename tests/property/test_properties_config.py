"""
Property-based tests for configuration loading and validation.

Tests Property 1: Configuration Validation
Validates: Requirements 8.1, 8.2
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import tempfile
import yaml
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from swellsight.utils.config import ConfigManager, SwellSightConfig, ModelConfig, TrainingConfig, DataConfig, SystemConfig

# Strategy for generating valid model configurations
model_config_strategy = st.builds(
    ModelConfig,
    depth_model_size=st.sampled_from(["small", "base", "large"]),
    depth_precision=st.sampled_from(["fp16", "fp32"]),
    backbone_model=st.sampled_from(["dinov2-base", "dinov2-large", "resnet50"]),
    freeze_backbone=st.booleans(),
    input_resolution=st.sampled_from([(224, 224), (518, 518), (640, 640)]),
    num_classes_direction=st.integers(min_value=2, max_value=5),
    num_classes_breaking=st.integers(min_value=2, max_value=5)
)

# Strategy for generating valid training configurations
training_config_strategy = st.builds(
    TrainingConfig,
    batch_size=st.integers(min_value=1, max_value=128),
    learning_rate=st.floats(min_value=1e-6, max_value=1e-1),
    num_epochs=st.integers(min_value=1, max_value=1000),
    weight_decay=st.floats(min_value=0, max_value=1e-2),
    gradient_clip_norm=st.floats(min_value=0.1, max_value=10.0),
    use_mixed_precision=st.booleans(),
    save_checkpoint_every=st.integers(min_value=1, max_value=100),
    validate_every=st.integers(min_value=1, max_value=50),
    early_stopping_patience=st.integers(min_value=1, max_value=100)
)

# Strategy for generating valid data configurations
data_config_strategy = st.builds(
    DataConfig,
    min_resolution=st.sampled_from([(640, 480), (1280, 720), (1920, 1080)]),
    max_resolution=st.sampled_from([(1920, 1080), (3840, 2160), (7680, 4320)]),
    quality_threshold=st.floats(min_value=0.0, max_value=1.0),
    augmentation_enabled=st.booleans(),
    synthetic_data_ratio=st.floats(min_value=0.0, max_value=1.0)
)

# Strategy for generating valid system configurations
system_config_strategy = st.builds(
    SystemConfig,
    use_gpu=st.booleans(),
    max_processing_time=st.floats(min_value=1.0, max_value=60.0),
    confidence_threshold=st.floats(min_value=0.0, max_value=1.0),
    log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR"]),
    output_dir=st.sampled_from(["outputs", "results", "data/outputs"])
)

# Strategy for generating complete SwellSight configurations
swellsight_config_strategy = st.builds(
    SwellSightConfig,
    model=model_config_strategy,
    training=training_config_strategy,
    data=data_config_strategy,
    system=system_config_strategy
)

class TestConfigurationValidation:
    """Property-based tests for configuration validation."""
    
    @given(config=swellsight_config_strategy)
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_property_1_configuration_validation(self, config):
        """
        Property 1: Configuration Validation
        
        For any valid SwellSightConfig, the ConfigManager should successfully
        load, validate, and save the configuration without errors.
        
        **Feature: wave-analysis-system, Property 1: Configuration Validation**
        **Validates: Requirements 8.1, 8.2**
        """
        # Ensure min_resolution <= max_resolution for data config
        assume(config.data.min_resolution[0] <= config.data.max_resolution[0])
        assume(config.data.min_resolution[1] <= config.data.max_resolution[1])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test YAML configuration
            yaml_path = Path(temp_dir) / "test_config.yaml"
            
            # Create a ConfigManager and set the config
            manager = ConfigManager()
            manager.config = config
            
            # Test validation
            is_valid = manager.validate_config()
            assert is_valid, f"Configuration validation failed for config: {config}"
            
            # Test saving to YAML
            manager.save_config(yaml_path)
            assert yaml_path.exists(), "Configuration file was not created"
            
            # Test loading from YAML
            loaded_manager = ConfigManager(yaml_path)
            loaded_config = loaded_manager.get_config()
            
            # Verify loaded configuration is valid
            assert loaded_manager.validate_config(), "Loaded configuration is invalid"
            
            # Test JSON configuration
            json_path = Path(temp_dir) / "test_config.json"
            manager.save_config(json_path)
            assert json_path.exists(), "JSON configuration file was not created"
            
            # Test loading from JSON
            json_manager = ConfigManager(json_path)
            json_config = json_manager.get_config()
            
            # Verify JSON loaded configuration is valid
            assert json_manager.validate_config(), "JSON loaded configuration is invalid"
    
    @given(
        depth_model_size=st.sampled_from(["invalid", "wrong", "bad"]),
        depth_precision=st.sampled_from(["invalid", "wrong", "bad"]),
        batch_size=st.integers(max_value=0),
        learning_rate=st.floats(max_value=0),
        quality_threshold=st.floats().filter(lambda x: x < 0 or x > 1),
        confidence_threshold=st.floats().filter(lambda x: x < 0 or x > 1)
    )
    def test_invalid_configuration_rejection(self, depth_model_size, depth_precision, 
                                           batch_size, learning_rate, quality_threshold, 
                                           confidence_threshold):
        """
        Test that invalid configurations are properly rejected.
        
        **Feature: wave-analysis-system, Property 1: Configuration Validation**
        **Validates: Requirements 8.1, 8.2**
        """
        # Create invalid configuration
        invalid_config = SwellSightConfig()
        invalid_config.model.depth_model_size = depth_model_size
        invalid_config.model.depth_precision = depth_precision
        invalid_config.training.batch_size = batch_size
        invalid_config.training.learning_rate = learning_rate
        invalid_config.data.quality_threshold = quality_threshold
        invalid_config.system.confidence_threshold = confidence_threshold
        
        manager = ConfigManager()
        manager.config = invalid_config
        
        # Validation should fail for invalid configuration
        is_valid = manager.validate_config()
        assert not is_valid, f"Invalid configuration was incorrectly validated as valid: {invalid_config}"
    
    def test_default_configuration_validity(self):
        """
        Test that the default configuration is always valid.
        
        **Feature: wave-analysis-system, Property 1: Configuration Validation**
        **Validates: Requirements 8.1, 8.2**
        """
        # Create manager with default configuration
        manager = ConfigManager()
        
        # Default configuration should always be valid
        assert manager.validate_config(), "Default configuration is invalid"
        
        # Should be able to get the configuration
        config = manager.get_config()
        assert config is not None, "Default configuration is None"
        assert isinstance(config, SwellSightConfig), "Default configuration is not SwellSightConfig"
    
    def test_nonexistent_config_file_handling(self):
        """
        Test handling of nonexistent configuration files.
        
        **Feature: wave-analysis-system, Property 1: Configuration Validation**
        **Validates: Requirements 8.1, 8.2**
        """
        nonexistent_path = Path("nonexistent_config.yaml")
        
        # Should not raise exception, should use defaults
        manager = ConfigManager(nonexistent_path)
        config = manager.get_config()
        
        assert config is not None, "Configuration should not be None for nonexistent file"
        assert isinstance(config, SwellSightConfig), "Should return default SwellSightConfig"
        assert manager.validate_config(), "Default configuration should be valid"
    
    @given(st.sampled_from([
        "invalid: yaml: content: {",
        "{ invalid json content",
        "random text that is not yaml or json",
        "key: value\n  invalid_indentation",
        "- list\n- with\n  - bad: indentation",
        '{"incomplete": "json"',
        "yaml: content\nwith: bad\n  indentation: here"
    ]))
    def test_corrupted_config_file_handling(self, corrupted_content):
        """
        Test handling of corrupted configuration files.
        
        **Feature: wave-analysis-system, Property 1: Configuration Validation**
        **Validates: Requirements 8.1, 8.2**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            corrupted_path = Path(temp_dir) / "corrupted_config.yaml"
            
            # Write corrupted content
            with open(corrupted_path, 'w', encoding='utf-8') as f:
                f.write(corrupted_content)
            
            # Should not raise exception, should use defaults
            manager = ConfigManager(corrupted_path)
            config = manager.get_config()
            
            assert config is not None, "Configuration should not be None for corrupted file"
            assert isinstance(config, SwellSightConfig), "Should return default SwellSightConfig"
            assert manager.validate_config(), "Default configuration should be valid"
    
    def _is_valid_yaml_or_json(self, content: str) -> bool:
        """Check if content is valid YAML or JSON."""
        try:
            yaml.safe_load(content)
            return True
        except:
            pass
        
        try:
            json.loads(content)
            return True
        except:
            pass
        
        return False

if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v"])