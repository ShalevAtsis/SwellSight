#!/usr/bin/env python3
"""
Property-Based Test for Configuration Loading with Defaults
Tests Property 36: Configuration Loading with Defaults
Validates Requirements 8.1: WHEN notebooks start, THE Notebook SHALL load configuration from a simple JSON file with sensible defaults
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    print("Hypothesis not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hypothesis"])
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True

from utils.config_manager import ConfigManager, load_config


@composite
def config_variations(draw):
    """Generate various configuration scenarios for testing"""
    # Base configuration structure - simplified for performance
    base_config = {
        "pipeline": {
            "name": draw(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=1, max_size=20)),
            "version": draw(st.text(alphabet=st.characters(whitelist_categories=('Nd', 'Po')), min_size=1, max_size=5)),
        },
        "processing": {
            "batch_size": draw(st.one_of(
                st.just("auto"),
                st.integers(min_value=1, max_value=16)
            )),
            "max_images": draw(st.integers(min_value=10, max_value=1000)),
            "quality_threshold": draw(st.floats(min_value=0.1, max_value=0.9)),
        },
        "models": {
            "mixed_precision": draw(st.booleans()),
        },
        "paths": {
            "data_dir": "./data",
            "output_dir": "./outputs",
        }
    }
    
    # Sometimes return partial config to test defaults (simplified)
    if draw(st.booleans()):
        # Remove one section to test default merging
        section_to_remove = draw(st.sampled_from(["processing", "models"]))
        if section_to_remove in base_config:
            del base_config[section_to_remove]
    
    return base_config


@composite
def invalid_config_variations(draw):
    """Generate invalid configuration scenarios for testing error handling"""
    config_type = draw(st.sampled_from([
        "missing_pipeline",
        "invalid_batch_size", 
        "invalid_quality_threshold",
        "missing_paths",
        "corrupted_json"
    ]))
    
    if config_type == "missing_pipeline":
        return {"processing": {"batch_size": 4}}
    elif config_type == "invalid_batch_size":
        return {
            "pipeline": {"name": "test", "version": "1.0"},
            "processing": {"batch_size": -1}
        }
    elif config_type == "invalid_quality_threshold":
        return {
            "pipeline": {"name": "test", "version": "1.0"},
            "processing": {"quality_threshold": 2.0}
        }
    elif config_type == "missing_paths":
        return {
            "pipeline": {"name": "test", "version": "1.0"},
            "processing": {"batch_size": 4}
        }
    else:  # corrupted_json
        return "invalid_json_content"


@given(config_variations())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_36_configuration_loading_with_defaults(config_data):
    """
    Feature: swellsight-pipeline-improvements, Property 36: Configuration Loading with Defaults
    
    Property: For any notebook startup, configuration should be loaded from JSON files with sensible defaults applied
    Validates: Requirements 8.1
    """
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        
        # Write test configuration
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        # Test configuration loading
        manager = ConfigManager(str(config_path))
        loaded_config = manager.load_config()
        
        # Property assertions: Configuration should always be loaded successfully with defaults
        assert loaded_config is not None, "Configuration should never be None"
        assert isinstance(loaded_config, dict), "Configuration should be a dictionary"
        
        # Essential sections should always be present (from defaults if missing)
        required_sections = ["pipeline", "processing", "models", "paths"]
        for section in required_sections:
            assert section in loaded_config, f"Required section '{section}' should be present (from defaults if needed)"
        
        # Pipeline section should have required fields
        assert "name" in loaded_config["pipeline"], "Pipeline name should be present"
        assert "version" in loaded_config["pipeline"], "Pipeline version should be present"
        assert isinstance(loaded_config["pipeline"]["name"], str), "Pipeline name should be string"
        assert isinstance(loaded_config["pipeline"]["version"], str), "Pipeline version should be string"
        
        # Processing section should have sensible defaults
        processing = loaded_config["processing"]
        assert "batch_size" in processing, "Batch size should be present"
        assert "quality_threshold" in processing, "Quality threshold should be present"
        
        # Batch size should be valid
        batch_size = processing["batch_size"]
        assert batch_size == "auto" or (isinstance(batch_size, int) and batch_size > 0), \
            f"Batch size should be 'auto' or positive integer, got {batch_size}"
        
        # Quality threshold should be in valid range
        quality_threshold = processing["quality_threshold"]
        assert isinstance(quality_threshold, (int, float)), "Quality threshold should be numeric"
        assert 0.0 <= quality_threshold <= 1.0, f"Quality threshold should be in [0,1], got {quality_threshold}"
        
        # Models section should have required fields
        models = loaded_config["models"]
        assert "mixed_precision" in models, "Mixed precision setting should be present"
        assert isinstance(models["mixed_precision"], bool), "Mixed precision should be boolean"
        
        # Paths section should have required paths
        paths = loaded_config["paths"]
        required_paths = ["data_dir", "output_dir", "checkpoint_dir"]
        for path_name in required_paths:
            assert path_name in paths, f"Required path '{path_name}' should be present"
            assert isinstance(paths[path_name], str), f"Path '{path_name}' should be string"


@given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=1, max_size=20))
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_36_missing_config_file_defaults(nonexistent_path):
    """
    Test that defaults are applied when config file doesn't exist
    Part of Property 36: Configuration Loading with Defaults
    """
    # Ensure path doesn't exist
    assume(not Path(nonexistent_path).exists())
    
    # Test loading from non-existent file
    manager = ConfigManager(nonexistent_path)
    loaded_config = manager.load_config()
    
    # Should load default configuration successfully
    assert loaded_config is not None, "Should load defaults when file doesn't exist"
    assert isinstance(loaded_config, dict), "Default configuration should be dictionary"
    
    # Should contain all required sections from defaults
    required_sections = ["pipeline", "processing", "models", "paths"]
    for section in required_sections:
        assert section in loaded_config, f"Default config should contain '{section}'"
    
    # Default values should be sensible (note: batch_size may be adapted from "auto" to integer)
    assert loaded_config["pipeline"]["name"] == "swellsight_pipeline", "Default pipeline name"
    
    # Batch size should be either "auto" or a positive integer (after hardware adaptation)
    batch_size = loaded_config["processing"]["batch_size"]
    assert batch_size == "auto" or (isinstance(batch_size, int) and batch_size > 0), \
        f"Default batch size should be 'auto' or positive integer, got {batch_size}"
    
    assert 0.0 <= loaded_config["processing"]["quality_threshold"] <= 1.0, "Default quality threshold in range"


@given(invalid_config_variations())
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_36_invalid_config_fallback_to_defaults(invalid_config):
    """
    Test that invalid configurations fall back to defaults gracefully
    Part of Property 36: Configuration Loading with Defaults
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "invalid_config.json"
        
        # Write invalid configuration
        try:
            if isinstance(invalid_config, str):
                # Write corrupted JSON
                with open(config_path, 'w') as f:
                    f.write(invalid_config)
            else:
                # Write invalid but parseable JSON
                with open(config_path, 'w') as f:
                    json.dump(invalid_config, f)
        except:
            # If we can't even write it, skip this test case
            assume(False)
        
        # Test loading invalid configuration
        manager = ConfigManager(str(config_path))
        loaded_config = manager.load_config()
        
        # Should fall back to defaults and still work
        assert loaded_config is not None, "Should fall back to defaults for invalid config"
        assert isinstance(loaded_config, dict), "Fallback should be dictionary"
        
        # Should contain all required sections from defaults
        required_sections = ["pipeline", "processing", "models", "paths"]
        for section in required_sections:
            assert section in loaded_config, f"Fallback config should contain '{section}'"
        
        # Validation should pass for fallback config
        is_valid, errors = manager.validate_config(loaded_config)
        assert is_valid, f"Fallback configuration should be valid, got errors: {errors}"


def test_property_36_convenience_function():
    """
    Test the convenience load_config function works with defaults
    Part of Property 36: Configuration Loading with Defaults
    """
    # Test with non-existent file (should use defaults)
    config = load_config("nonexistent_config.json")
    
    assert config is not None, "Convenience function should return config"
    assert isinstance(config, dict), "Should return dictionary"
    assert "pipeline" in config, "Should contain pipeline section"
    assert config["pipeline"]["name"] == "swellsight_pipeline", "Should use default name"
    
    # Batch size should be adapted but still valid
    batch_size = config["processing"]["batch_size"]
    assert batch_size == "auto" or (isinstance(batch_size, int) and batch_size > 0), \
        f"Batch size should be valid, got {batch_size}"


def run_property_tests():
    """Run all property tests for configuration loading"""
    print("Running Property-Based Tests for Configuration Loading...")
    print("=" * 60)
    
    try:
        # Run the property tests
        print("Testing Property 36: Configuration Loading with Defaults...")
        
        # Test 1: Valid configurations with defaults
        print("  ✓ Testing valid configurations with default merging...")
        test_property_36_configuration_loading_with_defaults()
        
        # Test 2: Missing config file defaults
        print("  ✓ Testing missing config file fallback to defaults...")
        test_property_36_missing_config_file_defaults()
        
        # Test 3: Invalid config fallback
        print("  ✓ Testing invalid config fallback to defaults...")
        test_property_36_invalid_config_fallback_to_defaults()
        
        # Test 4: Convenience function
        print("  ✓ Testing convenience function with defaults...")
        test_property_36_convenience_function()
        
        print("\n🎉 All property tests passed!")
        print("Property 36 (Configuration Loading with Defaults) validated successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Property test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_tests()
    exit(0 if success else 1)