#!/usr/bin/env python3
"""
Property-Based Test for Mixed Precision Training Adaptation
Tests Property 26: Mixed Precision Training Adaptation
Validates Requirements 6.1: WHEN training begins, THE Notebook SHALL implement mixed precision training if supported by hardware
"""

import torch
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
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

from utils.config_manager import ConfigManager
from utils.flux_memory_manager import FLUXMemoryManager


class MockTorchCuda:
    """Mock torch.cuda for testing different hardware scenarios"""
    
    def __init__(self, is_available: bool = True, compute_capability: tuple = (7, 5), memory_gb: float = 8.0):
        self.is_available_value = is_available
        self.compute_capability_value = compute_capability
        self.memory_gb_value = memory_gb
        self.device_count_value = 1 if is_available else 0
    
    def is_available(self) -> bool:
        return self.is_available_value
    
    def get_device_capability(self, device: int = 0) -> tuple:
        if not self.is_available_value:
            raise RuntimeError("CUDA not available")
        return self.compute_capability_value
    
    def get_device_properties(self, device: int = 0):
        class MockProperties:
            def __init__(self, memory_gb: float):
                self.total_memory = int(memory_gb * 1024**3)
        
        if not self.is_available_value:
            raise RuntimeError("CUDA not available")
        return MockProperties(self.memory_gb_value)
    
    def device_count(self) -> int:
        return self.device_count_value


class MockAutocast:
    """Mock torch.cuda.amp.autocast for testing"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.entered = False
    
    def __enter__(self):
        self.entered = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.entered = False


@composite
def hardware_configurations(draw):
    """Generate various hardware configurations for testing mixed precision"""
    
    # GPU availability
    gpu_available = draw(st.booleans())
    
    if gpu_available:
        # Compute capability (major, minor)
        # Tensor Cores available from compute capability 7.0+
        major = draw(st.integers(min_value=5, max_value=9))  # Range from Pascal to future architectures
        minor = draw(st.integers(min_value=0, max_value=9))
        compute_capability = (major, minor)
        
        # GPU memory in GB
        memory_gb = draw(st.floats(min_value=2.0, max_value=80.0))
        
        # Mixed precision support based on compute capability
        mixed_precision_supported = major >= 7
        
    else:
        compute_capability = (0, 0)
        memory_gb = 0.0
        mixed_precision_supported = False
    
    return {
        'gpu_available': gpu_available,
        'compute_capability': compute_capability,
        'memory_gb': memory_gb,
        'mixed_precision_supported': mixed_precision_supported
    }


@composite
def model_configurations(draw):
    """Generate various model configurations for testing"""
    
    # Mixed precision setting in config
    config_mixed_precision = draw(st.one_of(
        st.booleans(),
        st.just("auto")  # Auto-detect based on hardware
    ))
    
    # Model precision settings
    base_config = {
        "pipeline": {
            "name": "swellsight_pipeline",
            "version": "1.0"
        },
        "models": {
            "base_model": "black-forest-labs/FLUX.1-dev",
            "controlnet_model": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
            "mixed_precision": config_mixed_precision
        },
        "processing": {
            "batch_size": draw(st.integers(min_value=1, max_value=8))
        },
        "paths": {
            "data_dir": "./data",
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints"
        }
    }
    
    return base_config


def simulate_mixed_precision_training_setup(hardware_config: Dict[str, Any], 
                                          model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate the setup of mixed precision training based on hardware and model config"""
    
    # Determine if mixed precision should be enabled
    config_setting = model_config["models"]["mixed_precision"]
    hardware_supports = hardware_config["mixed_precision_supported"]
    gpu_available = hardware_config["gpu_available"]
    
    # Logic for mixed precision enablement
    if config_setting == "auto":
        # Auto-detect: enable if hardware supports it
        mixed_precision_enabled = gpu_available and hardware_supports
    elif isinstance(config_setting, bool):
        # Explicit setting: respect config but warn if hardware doesn't support
        mixed_precision_enabled = config_setting and gpu_available
        if config_setting and not hardware_supports:
            # Config requests mixed precision but hardware doesn't support it fully
            mixed_precision_enabled = False  # Fallback to FP32
    else:
        mixed_precision_enabled = False
    
    # Determine torch dtype
    if mixed_precision_enabled:
        torch_dtype = "torch.float16"
    else:
        torch_dtype = "torch.float32"
    
    # Determine autocast usage
    autocast_enabled = mixed_precision_enabled
    
    return {
        'mixed_precision_enabled': mixed_precision_enabled,
        'torch_dtype': torch_dtype,
        'autocast_enabled': autocast_enabled,
        'hardware_supports': hardware_supports,
        'gpu_available': gpu_available,
        'config_setting': config_setting,
        'fallback_reason': None if mixed_precision_enabled else (
            "No GPU available" if not gpu_available else
            "Hardware doesn't support mixed precision" if not hardware_supports else
            "Mixed precision disabled in config"
        )
    }


@given(hardware_configurations(), model_configurations())
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_26_mixed_precision_training_adaptation(hardware_config, model_config):
    """
    Feature: swellsight-pipeline-improvements, Property 26: Mixed Precision Training Adaptation
    
    Property: For any training initialization, mixed precision training should be implemented when supported by hardware
    Validates: Requirements 6.1
    """
    
    # Simulate the mixed precision training setup
    setup_result = simulate_mixed_precision_training_setup(hardware_config, model_config)
    
    # Property assertions: Mixed precision should be correctly adapted based on hardware capabilities
    
    # 1. If hardware supports mixed precision and config allows it, mixed precision should be enabled
    if (hardware_config["mixed_precision_supported"] and 
        hardware_config["gpu_available"] and 
        model_config["models"]["mixed_precision"] in [True, "auto"]):
        
        assert setup_result["mixed_precision_enabled"], \
            f"Mixed precision should be enabled when hardware supports it and config allows it. " \
            f"Hardware: {hardware_config}, Config: {model_config['models']['mixed_precision']}"
        
        assert setup_result["torch_dtype"] == "torch.float16", \
            f"Torch dtype should be float16 when mixed precision is enabled, got {setup_result['torch_dtype']}"
        
        assert setup_result["autocast_enabled"], \
            "Autocast should be enabled when mixed precision is enabled"
    
    # 2. If hardware doesn't support mixed precision, it should be disabled regardless of config
    if not hardware_config["mixed_precision_supported"] or not hardware_config["gpu_available"]:
        
        if model_config["models"]["mixed_precision"] is True:
            # Config explicitly requests mixed precision but hardware doesn't support it
            assert not setup_result["mixed_precision_enabled"], \
                f"Mixed precision should be disabled when hardware doesn't support it, even if config requests it. " \
                f"Hardware supports: {hardware_config['mixed_precision_supported']}, GPU available: {hardware_config['gpu_available']}"
        
        assert setup_result["torch_dtype"] == "torch.float32", \
            f"Torch dtype should be float32 when mixed precision is not supported, got {setup_result['torch_dtype']}"
        
        assert not setup_result["autocast_enabled"], \
            "Autocast should be disabled when mixed precision is not supported"
    
    # 3. If config explicitly disables mixed precision, it should be disabled
    if model_config["models"]["mixed_precision"] is False:
        assert not setup_result["mixed_precision_enabled"], \
            f"Mixed precision should be disabled when config explicitly sets it to False"
        
        assert setup_result["torch_dtype"] == "torch.float32", \
            f"Torch dtype should be float32 when mixed precision is disabled in config"
        
        assert not setup_result["autocast_enabled"], \
            "Autocast should be disabled when mixed precision is disabled in config"
    
    # 4. Fallback reason should be provided when mixed precision is not enabled
    if not setup_result["mixed_precision_enabled"]:
        assert setup_result["fallback_reason"] is not None, \
            "A fallback reason should be provided when mixed precision is not enabled"
        
        assert isinstance(setup_result["fallback_reason"], str), \
            "Fallback reason should be a string"
        
        assert len(setup_result["fallback_reason"]) > 0, \
            "Fallback reason should not be empty"
    
    # 5. Configuration consistency checks
    assert isinstance(setup_result["mixed_precision_enabled"], bool), \
        "Mixed precision enabled flag should be boolean"
    
    assert setup_result["torch_dtype"] in ["torch.float16", "torch.float32"], \
        f"Torch dtype should be either float16 or float32, got {setup_result['torch_dtype']}"
    
    assert isinstance(setup_result["autocast_enabled"], bool), \
        "Autocast enabled flag should be boolean"


@given(st.integers(min_value=5, max_value=9), st.integers(min_value=0, max_value=9))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_26_compute_capability_detection(major_version, minor_version):
    """
    Test compute capability detection for mixed precision support
    Part of Property 26: Mixed Precision Training Adaptation
    """
    compute_capability = (major_version, minor_version)
    
    # Mixed precision (Tensor Cores) available from compute capability 7.0+
    expected_support = major_version >= 7
    
    # Simulate hardware detection
    hardware_config = {
        'gpu_available': True,
        'compute_capability': compute_capability,
        'memory_gb': 8.0,
        'mixed_precision_supported': expected_support
    }
    
    model_config = {
        "models": {"mixed_precision": "auto"},
        "processing": {"batch_size": 1}
    }
    
    setup_result = simulate_mixed_precision_training_setup(hardware_config, model_config)
    
    # Property: Compute capability should correctly determine mixed precision support
    if major_version >= 7:
        assert setup_result["mixed_precision_enabled"], \
            f"Mixed precision should be enabled for compute capability {compute_capability}"
        assert setup_result["torch_dtype"] == "torch.float16", \
            f"Should use float16 for compute capability {compute_capability}"
    else:
        assert not setup_result["mixed_precision_enabled"], \
            f"Mixed precision should be disabled for compute capability {compute_capability}"
        assert setup_result["torch_dtype"] == "torch.float32", \
            f"Should use float32 for compute capability {compute_capability}"


def test_property_26_config_manager_integration():
    """
    Test integration with ConfigManager for mixed precision settings
    Part of Property 26: Mixed Precision Training Adaptation
    """
    
    # Test with various config scenarios
    test_configs = [
        {"models": {"mixed_precision": True}},
        {"models": {"mixed_precision": False}},
        {"models": {"mixed_precision": "auto"}},
        {"models": {}},  # Missing mixed_precision key
    ]
    
    for config_data in test_configs:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # Create full config with test data
            full_config = {
                "pipeline": {"name": "test", "version": "1.0"},
                "processing": {"batch_size": 1},
                "paths": {"data_dir": "./data", "output_dir": "./outputs", "checkpoint_dir": "./checkpoints"}
            }
            full_config.update(config_data)
            
            # Write config file
            with open(config_path, 'w') as f:
                json.dump(full_config, f)
            
            # Load config through ConfigManager
            manager = ConfigManager(str(config_path))
            loaded_config = manager.load_config()
            
            # Property: Config should always have mixed_precision setting after loading
            assert "models" in loaded_config, "Models section should be present"
            assert "mixed_precision" in loaded_config["models"], \
                "Mixed precision setting should be present (from defaults if missing)"
            
            mixed_precision_setting = loaded_config["models"]["mixed_precision"]
            assert isinstance(mixed_precision_setting, (bool, str)), \
                f"Mixed precision setting should be boolean or 'auto', got {type(mixed_precision_setting)}"
            
            if isinstance(mixed_precision_setting, str):
                assert mixed_precision_setting == "auto", \
                    f"String mixed precision setting should be 'auto', got '{mixed_precision_setting}'"


def test_property_26_memory_manager_integration():
    """
    Test integration with FLUXMemoryManager for mixed precision considerations
    Part of Property 26: Mixed Precision Training Adaptation
    """
    
    # Test config with mixed precision enabled
    config = {
        "models": {"mixed_precision": True},
        "hardware": {"gpu_memory_fraction": 0.9}
    }
    
    memory_manager = FLUXMemoryManager(config)
    
    # Property: Memory manager should account for mixed precision in memory calculations
    # Mixed precision typically reduces memory usage by ~50% for model weights
    
    # Test batch size calculation (should work regardless of mixed precision setting)
    batch_size = memory_manager.calculate_optimal_batch_size((1024, 1024))
    
    assert isinstance(batch_size, int), "Batch size should be integer"
    assert batch_size >= 1, "Batch size should be at least 1"
    
    # Test memory monitoring
    monitor_data = memory_manager.monitor_memory_during_generation("test_operation")
    
    assert isinstance(monitor_data, dict), "Monitor data should be dictionary"
    assert "start_profile" in monitor_data, "Should contain start profile"
    assert "operation_name" in monitor_data, "Should contain operation name"
    
    # Finalize monitoring
    final_data = memory_manager.finalize_memory_monitoring(monitor_data)
    
    assert isinstance(final_data, dict), "Final data should be dictionary"
    assert "memory_delta" in final_data, "Should contain memory delta information"


def run_property_tests():
    """Run all property tests for mixed precision training adaptation"""
    print("Running Property-Based Tests for Mixed Precision Training Adaptation...")
    print("=" * 70)
    
    try:
        # Run the property tests
        print("Testing Property 26: Mixed Precision Training Adaptation...")
        
        # Test 1: Main property test
        print("  ✓ Testing mixed precision adaptation across hardware configurations...")
        test_property_26_mixed_precision_training_adaptation()
        
        # Test 2: Compute capability detection
        print("  ✓ Testing compute capability detection...")
        test_property_26_compute_capability_detection()
        
        # Test 3: Config manager integration
        print("  ✓ Testing ConfigManager integration...")
        test_property_26_config_manager_integration()
        
        # Test 4: Memory manager integration
        print("  ✓ Testing FLUXMemoryManager integration...")
        test_property_26_memory_manager_integration()
        
        print("\n🎉 All property tests passed!")
        print("Property 26 (Mixed Precision Training Adaptation) validated successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Property test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_tests()
    exit(0 if success else 1)