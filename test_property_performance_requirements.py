#!/usr/bin/env python3
"""
Property-Based Test for Performance Requirements
Feature: swellsight-pipeline-improvements, Property 19: Real-Time Performance, Property 20: End-to-End Processing Speed
Validates: Requirements 7.5, 8.1, 8.2
"""

import sys
import pytest
import numpy as np
import torch
import time
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, example, assume, HealthCheck

# Add src to path
sys.path.append('src')

from swellsight.utils.performance import PerformanceOptimizer, OptimizationConfig, PerformanceMetrics
from swellsight.utils.hardware import HardwareManager

class TestPerformanceRequirements:
    """Property-based tests for performance requirements"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Disable torch.compile for Windows compatibility
        self.config = OptimizationConfig(
            target_latency_ms=200.0,
            enable_mixed_precision=False,
            enable_torch_compile=False,
            batch_size=1
        )
        self.optimizer = PerformanceOptimizer(self.config)
        self.hardware_manager = HardwareManager()
    
    def test_basic_real_time_performance(self):
        """
        Basic test for real-time performance requirement
        
        Feature: swellsight-pipeline-improvements, Property 19: Real-Time Performance
        Validates: Requirements 7.5, 8.1, 8.2
        """
        # Create a simple model that should be fast
        class FastModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = FastModel()
        optimized_model = self.optimizer.optimize_model(model)
        
        # Warmup
        self.optimizer.warmup_model(optimized_model, (10,))
        
        # Test single inference
        input_tensor = torch.randn(1, 10)
        output, metrics = self.optimizer.optimize_inference(optimized_model, input_tensor)
        
        # Should meet real-time requirement
        assert metrics.total_time_ms <= self.config.target_latency_ms, \
            f"Performance requirement not met: {metrics.total_time_ms:.2f}ms > {self.config.target_latency_ms}ms"
        
        # Should have reasonable throughput
        assert metrics.throughput_fps > 0, "Throughput should be positive"
        
        print(f"✓ Basic real-time test passed - Time: {metrics.total_time_ms:.2f}ms, FPS: {metrics.throughput_fps:.1f}")
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        input_size=st.integers(min_value=10, max_value=50),
        target_latency=st.floats(min_value=100.0, max_value=300.0)
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    @example(batch_size=1, input_size=32, target_latency=200.0)
    def test_real_time_performance_property(self, batch_size: int, input_size: int, target_latency: float):
        """
        Property: For any reasonable input configuration, the system should 
        provide consistent performance metrics and attempt to meet latency targets
        
        Feature: swellsight-pipeline-improvements, Property 19: Real-Time Performance
        Validates: Requirements 7.5, 8.1, 8.2
        """
        # Ensure reasonable parameters
        assume(batch_size >= 1 and batch_size <= 8)
        assume(input_size >= 10 and input_size <= 100)
        assume(target_latency >= 50.0 and target_latency <= 500.0)
        
        # Create optimizer with target latency
        config = OptimizationConfig(
            target_latency_ms=target_latency,
            enable_mixed_precision=False,
            enable_torch_compile=False,
            batch_size=batch_size
        )
        optimizer = PerformanceOptimizer(config)
        
        # Create model with variable size
        class VariableModel(torch.nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.linear1 = torch.nn.Linear(input_size, input_size // 2 + 1)
                self.linear2 = torch.nn.Linear(input_size // 2 + 1, 1)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                return self.linear2(x)
        
        model = VariableModel(input_size)
        optimized_model = optimizer.optimize_model(model)
        
        # Warmup
        optimizer.warmup_model(optimized_model, (input_size,))
        
        # Test inference
        input_tensor = torch.randn(batch_size, input_size)
        output, metrics = optimizer.optimize_inference(optimized_model, input_tensor)
        
        # Property assertions
        # 1. Metrics should be valid
        assert isinstance(metrics, PerformanceMetrics), "Should return valid PerformanceMetrics"
        assert metrics.total_time_ms > 0, "Total time should be positive"
        assert metrics.inference_time_ms >= 0, "Inference time should be non-negative"
        assert metrics.preprocessing_time_ms >= 0, "Preprocessing time should be non-negative"
        assert metrics.postprocessing_time_ms >= 0, "Postprocessing time should be non-negative"
        assert metrics.throughput_fps > 0, "Throughput should be positive"
        
        # 2. Time components should sum approximately to total
        component_sum = metrics.inference_time_ms + metrics.preprocessing_time_ms + metrics.postprocessing_time_ms
        assert abs(component_sum - metrics.total_time_ms) <= 5.0, \
            f"Time components should sum to total: {component_sum:.2f} vs {metrics.total_time_ms:.2f}"
        
        # 3. Throughput should be consistent with total time
        expected_fps = 1000.0 / metrics.total_time_ms
        assert abs(metrics.throughput_fps - expected_fps) <= 1.0, \
            f"Throughput should match total time: {metrics.throughput_fps:.1f} vs {expected_fps:.1f}"
        
        # 4. Output should have correct shape
        assert output is not None, "Should produce output"
        if isinstance(output, torch.Tensor):
            assert output.shape[0] == batch_size, f"Output batch size should match input: {output.shape[0]} vs {batch_size}"
    
    def test_end_to_end_processing_speed(self):
        """
        Test end-to-end processing speed requirements
        
        Feature: swellsight-pipeline-improvements, Property 20: End-to-End Processing Speed
        Validates: Requirements 7.5, 8.1, 8.2
        """
        # Simulate end-to-end pipeline with multiple stages
        class PipelineStage(torch.nn.Module):
            def __init__(self, name, processing_time_ms=10):
                super().__init__()
                self.name = name
                self.processing_time_ms = processing_time_ms
                self.linear = torch.nn.Linear(64, 64)
            
            def forward(self, x):
                # Simulate processing time (in a real scenario this would be actual computation)
                return self.linear(x)
        
        # Create pipeline stages
        depth_stage = PipelineStage("depth_extraction", 50)
        analysis_stage = PipelineStage("wave_analysis", 100)
        
        # Optimize stages
        optimized_depth = self.optimizer.optimize_model(depth_stage)
        optimized_analysis = self.optimizer.optimize_model(analysis_stage)
        
        # Warmup stages
        self.optimizer.warmup_model(optimized_depth, (64,))
        self.optimizer.warmup_model(optimized_analysis, (64,))
        
        # Test end-to-end processing
        input_tensor = torch.randn(1, 64)
        
        # Stage 1: Depth extraction
        depth_output, depth_metrics = self.optimizer.optimize_inference(optimized_depth, input_tensor)
        
        # Stage 2: Wave analysis
        analysis_output, analysis_metrics = self.optimizer.optimize_inference(optimized_analysis, depth_output)
        
        # Calculate total end-to-end time
        total_time_ms = depth_metrics.total_time_ms + analysis_metrics.total_time_ms
        
        # End-to-end requirements
        # Should complete within reasonable time for real-time processing
        max_end_to_end_time = 300.0  # 300ms for full pipeline
        assert total_time_ms <= max_end_to_end_time, \
            f"End-to-end processing too slow: {total_time_ms:.2f}ms > {max_end_to_end_time}ms"
        
        # Should maintain throughput
        end_to_end_fps = 1000.0 / total_time_ms
        min_required_fps = 3.0  # At least 3 FPS for practical use
        assert end_to_end_fps >= min_required_fps, \
            f"End-to-end throughput too low: {end_to_end_fps:.1f} FPS < {min_required_fps} FPS"
        
        print(f"✓ End-to-end test passed - Total: {total_time_ms:.2f}ms, FPS: {end_to_end_fps:.1f}")
    
    def test_performance_consistency_property(self):
        """
        Property: Performance should be consistent across multiple runs
        
        Feature: swellsight-pipeline-improvements, Property 19: Real-Time Performance
        Validates: Requirements 7.5, 8.1, 8.2
        """
        # Create consistent model
        class ConsistentModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 16)
            
            def forward(self, x):
                return torch.relu(self.linear(x))
        
        model = ConsistentModel()
        optimized_model = self.optimizer.optimize_model(model)
        
        # Warmup
        self.optimizer.warmup_model(optimized_model, (32,))
        
        # Run multiple inferences
        times = []
        for i in range(10):
            input_tensor = torch.randn(1, 32)
            output, metrics = self.optimizer.optimize_inference(optimized_model, input_tensor)
            times.append(metrics.total_time_ms)
        
        # Check consistency
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Standard deviation should be reasonable (less than 50% of mean)
        max_std_ratio = 0.5
        std_ratio = std_time / mean_time if mean_time > 0 else 0
        assert std_ratio <= max_std_ratio, \
            f"Performance too inconsistent: std/mean = {std_ratio:.2f} > {max_std_ratio}"
        
        # All times should be within reasonable bounds
        max_time = max(times)
        min_time = min(times)
        time_range_ratio = (max_time - min_time) / mean_time if mean_time > 0 else 0
        max_range_ratio = 1.2  # Allow some variation in timing
        assert time_range_ratio <= max_range_ratio, \
            f"Performance range too large: range/mean = {time_range_ratio:.2f} > {max_range_ratio}"
        
        print(f"✓ Consistency test passed - Mean: {mean_time:.2f}ms, Std: {std_time:.2f}ms")
    
    def test_hardware_adaptation_property(self):
        """
        Property: Performance optimization should adapt to available hardware
        
        Feature: swellsight-pipeline-improvements, Property 19: Real-Time Performance
        Validates: Requirements 8.3, 8.4
        """
        # Test hardware detection
        hardware_info = self.hardware_manager.get_system_info()
        
        assert "hardware_info" in hardware_info, "Should provide hardware information"
        assert "cuda_available" in hardware_info, "Should detect CUDA availability"
        assert "preferred_device" in hardware_info, "Should select preferred device"
        
        # Test optimal batch size calculation
        # Small model, small input
        small_batch = self.hardware_manager.get_optimal_batch_size(100, 10)  # 100MB model, 10MB input
        
        # Large model, large input
        large_batch = self.hardware_manager.get_optimal_batch_size(2000, 100)  # 2GB model, 100MB input
        
        # Properties
        assert small_batch >= 1, "Should always allow at least batch size 1"
        assert large_batch >= 1, "Should always allow at least batch size 1"
        assert small_batch >= large_batch, "Smaller models should allow larger batch sizes"
        
        # Test memory requirements checking
        # Should handle reasonable memory requirements
        assert self.hardware_manager.check_memory_requirements(0.1), "Should handle small memory requirements"
        
        # Should reject unreasonable memory requirements
        unreasonable_memory = self.hardware_manager.hardware_info.memory_total_gb * 2  # 2x total memory
        assert not self.hardware_manager.check_memory_requirements(unreasonable_memory), \
            "Should reject unreasonable memory requirements"
        
        print(f"✓ Hardware adaptation test passed - Small batch: {small_batch}, Large batch: {large_batch}")
    
    def test_performance_monitoring_property(self):
        """
        Property: Performance monitoring should provide accurate statistics
        
        Feature: swellsight-pipeline-improvements, Property 20: End-to-End Processing Speed
        Validates: Requirements 7.5, 8.1, 8.2
        """
        # Clear any existing history
        self.optimizer.clear_performance_history()
        
        # Initially no stats
        stats = self.optimizer.get_performance_stats()
        assert stats == {}, "Should have no stats initially"
        
        # Create model and run inferences
        class MonitoredModel(torch.nn.Module):
            def forward(self, x):
                return x * 0.5  # Simple operation
        
        model = MonitoredModel()
        optimized_model = self.optimizer.optimize_model(model)
        
        # Run multiple inferences
        num_runs = 5
        for i in range(num_runs):
            input_tensor = torch.randn(1, 20)
            _, metrics = self.optimizer.optimize_inference(optimized_model, input_tensor)
        
        # Check statistics
        stats = self.optimizer.get_performance_stats()
        
        # Should have all required statistics
        required_stats = [
            "avg_total_time_ms", "min_total_time_ms", "max_total_time_ms",
            "std_total_time_ms", "avg_inference_time_ms", "avg_throughput_fps",
            "target_met_percentage", "num_samples"
        ]
        
        for stat in required_stats:
            assert stat in stats, f"Should provide {stat} statistic"
        
        # Statistics should be reasonable
        assert stats["num_samples"] == num_runs, f"Should track correct number of samples: {stats['num_samples']} vs {num_runs}"
        assert stats["avg_total_time_ms"] > 0, "Average time should be positive"
        assert stats["min_total_time_ms"] <= stats["avg_total_time_ms"], "Min should be <= average"
        assert stats["max_total_time_ms"] >= stats["avg_total_time_ms"], "Max should be >= average"
        assert stats["std_total_time_ms"] >= 0, "Standard deviation should be non-negative"
        assert 0 <= stats["target_met_percentage"] <= 100, "Target met percentage should be 0-100"
        
        print(f"✓ Monitoring test passed - Avg: {stats['avg_total_time_ms']:.2f}ms, Target met: {stats['target_met_percentage']:.1f}%")


if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v", "--tb=short"])