#!/usr/bin/env python3
"""
Simple test for performance optimization functionality.
Tests that the performance optimization system works correctly.
"""

import sys
import numpy as np
import torch
import time
from unittest.mock import Mock, patch

# Add src to path
sys.path.append('src')

from swellsight.utils.performance import PerformanceOptimizer, OptimizationConfig, PerformanceMetrics
from swellsight.utils.hardware import HardwareManager

def test_performance_optimizer_basic():
    """Test basic performance optimizer functionality."""
    print("Testing PerformanceOptimizer basic functionality...")
    
    # Create optimizer
    config = OptimizationConfig(target_latency_ms=200.0, enable_mixed_precision=False, enable_torch_compile=False)
    optimizer = PerformanceOptimizer(config)
    
    # Create a simple mock model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    
    # Test model optimization
    optimized_model = optimizer.optimize_model(model)
    assert optimized_model is not None
    
    # Test warmup
    optimizer.warmup_model(optimized_model, (10,))
    
    # Test inference
    input_tensor = torch.randn(1, 10)
    output, metrics = optimizer.optimize_inference(optimized_model, input_tensor)
    
    assert output is not None
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_time_ms > 0
    assert metrics.throughput_fps > 0
    
    print(f"✓ Basic test passed - Total time: {metrics.total_time_ms:.2f}ms")

def test_performance_measurement():
    """Test performance measurement context manager."""
    print("Testing performance measurement...")
    
    optimizer = PerformanceOptimizer()
    
    with optimizer.measure_time("test_operation") as timing:
        time.sleep(0.01)  # Simulate 10ms operation
    
    assert "test_operation_time_ms" in timing
    assert timing["test_operation_time_ms"] >= 10.0  # Should be at least 10ms
    
    print(f"✓ Measurement test passed - Measured time: {timing['test_operation_time_ms']:.2f}ms")

def test_hardware_integration():
    """Test hardware manager integration."""
    print("Testing hardware manager integration...")
    
    hardware_manager = HardwareManager()
    system_info = hardware_manager.get_system_info()
    
    assert "hardware_info" in system_info
    assert "pytorch_version" in system_info
    assert "cuda_available" in system_info
    
    # Test optimal batch size calculation
    batch_size = hardware_manager.get_optimal_batch_size(1000, 100)  # 1GB model, 100MB input
    assert batch_size >= 1
    
    print(f"✓ Hardware integration test passed - Optimal batch size: {batch_size}")

def test_performance_stats():
    """Test performance statistics collection."""
    print("Testing performance statistics...")
    
    # Disable torch.compile for Windows compatibility
    config = OptimizationConfig(enable_torch_compile=False)
    optimizer = PerformanceOptimizer(config)
    
    # Initially no stats
    stats = optimizer.get_performance_stats()
    assert stats == {}
    
    # Create mock model and run some inferences
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2
    
    model = SimpleModel()
    optimized_model = optimizer.optimize_model(model)
    
    # Run multiple inferences to build history
    for i in range(5):
        input_tensor = torch.randn(1, 10)
        _, metrics = optimizer.optimize_inference(optimized_model, input_tensor)
    
    # Check stats
    stats = optimizer.get_performance_stats()
    assert "avg_total_time_ms" in stats
    assert "num_samples" in stats
    assert stats["num_samples"] == 5
    
    print(f"✓ Statistics test passed - Avg time: {stats['avg_total_time_ms']:.2f}ms")

def main():
    """Run all performance optimization tests."""
    print("Running performance optimization tests...\n")
    
    try:
        test_performance_optimizer_basic()
        test_performance_measurement()
        test_hardware_integration()
        test_performance_stats()
        
        print("\n✅ All performance optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)