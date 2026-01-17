"""
Property-based tests for performance benchmarking consistency.

Tests that performance benchmarking produces consistent and valid results
across different scenarios and hardware configurations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch

from src.swellsight.evaluation.benchmarks import PerformanceBenchmarker, HardwareConfig, PerformanceBenchmark


class SimpleTestModel(nn.Module):
    """Simple test model for benchmarking."""
    
    def __init__(self, input_channels=3, num_classes=3):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_height = nn.Linear(16, 1)
        self.fc_direction = nn.Linear(16, num_classes)
        self.fc_breaking = nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        height = self.fc_height(x)
        direction = self.fc_direction(x)
        breaking = self.fc_breaking(x)
        return {
            "wave_height": height,
            "wave_direction": direction,
            "breaking_type": breaking
        }


class TestPerformanceBenchmarkingConsistency:
    """Property tests for performance benchmarking consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for consistent testing
        self.model = SimpleTestModel()
        self.benchmarker = PerformanceBenchmarker(self.model, self.device)
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        height=st.integers(min_value=32, max_value=128),
        width=st.integers(min_value=32, max_value=128),
        channels=st.integers(min_value=3, max_value=4)
    )
    @settings(max_examples=20, deadline=10000)
    def test_inference_speed_consistency(self, batch_size, height, width, channels):
        """Test that inference speed benchmarking is consistent."""
        # Create input tensor
        input_tensor = torch.randn(batch_size, channels, height, width)
        
        # Run benchmark multiple times
        times = []
        for _ in range(3):
            avg_time = self.benchmarker.benchmark_inference_speed(
                input_tensor, num_warmup=2, num_iterations=5
            )
            times.append(avg_time)
        
        # Property 1: All times should be positive
        for time_ms in times:
            assert time_ms > 0, f"Inference time should be positive, got {time_ms}"
        
        # Property 2: Times should be reasonably consistent (within 50% variation)
        if len(times) > 1:
            mean_time = np.mean(times)
            max_deviation = max(abs(t - mean_time) for t in times)
            relative_deviation = max_deviation / mean_time if mean_time > 0 else 0
            
            assert relative_deviation < 0.5, \
                f"Inference times should be consistent, got relative deviation {relative_deviation:.3f}"
        
        # Property 3: Larger inputs should generally take longer (with some tolerance)
        small_input = torch.randn(1, 3, 32, 32)
        large_input = torch.randn(batch_size, channels, height, width)
        
        small_time = self.benchmarker.benchmark_inference_speed(
            small_input, num_warmup=1, num_iterations=3
        )
        large_time = self.benchmarker.benchmark_inference_speed(
            large_input, num_warmup=1, num_iterations=3
        )
        
        # Allow some tolerance for measurement noise
        input_size_ratio = (batch_size * channels * height * width) / (1 * 3 * 32 * 32)
        if input_size_ratio > 2:  # Only check if significantly larger
            assert large_time >= small_time * 0.5, \
                f"Larger input should take at least half the time of smaller input, got {large_time} vs {small_time}"
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        channels=st.integers(min_value=3, max_value=4),
        size=st.integers(min_value=32, max_value=96)
    )
    @settings(max_examples=15, deadline=8000)
    def test_memory_usage_properties(self, batch_size, channels, size):
        """Test properties of memory usage measurement."""
        input_tensor = torch.randn(batch_size, channels, size, size)
        
        # Measure memory usage
        memory_stats = self.benchmarker.measure_memory_usage(input_tensor)
        
        # Property 1: All memory measurements should be non-negative
        assert memory_stats["baseline_memory_mb"] >= 0, \
            f"Baseline memory should be non-negative, got {memory_stats['baseline_memory_mb']}"
        assert memory_stats["peak_memory_mb"] >= 0, \
            f"Peak memory should be non-negative, got {memory_stats['peak_memory_mb']}"
        assert memory_stats["current_memory_mb"] >= 0, \
            f"Current memory should be non-negative, got {memory_stats['current_memory_mb']}"
        
        # Property 2: Peak memory should be >= baseline memory
        assert memory_stats["peak_memory_mb"] >= memory_stats["baseline_memory_mb"], \
            f"Peak memory ({memory_stats['peak_memory_mb']}) should be >= baseline ({memory_stats['baseline_memory_mb']})"
        
        # Property 3: Memory increase should be non-negative
        assert memory_stats["memory_increase_mb"] >= 0, \
            f"Memory increase should be non-negative, got {memory_stats['memory_increase_mb']}"
        
        # Property 4: GPU utilization should be between 0 and 100 (for CPU it should be 0)
        assert 0 <= memory_stats["gpu_utilization"] <= 100, \
            f"GPU utilization should be 0-100%, got {memory_stats['gpu_utilization']}"
        
        # For CPU device, GPU utilization should be 0
        if self.device.type == "cpu":
            assert memory_stats["gpu_utilization"] == 0, \
                f"CPU device should have 0% GPU utilization, got {memory_stats['gpu_utilization']}"
    
    @given(
        batch_sizes=st.lists(
            st.integers(min_value=1, max_value=8),
            min_size=2, max_size=5, unique=True
        ).map(sorted),
        channels=st.integers(min_value=3, max_value=4),
        size=st.integers(min_value=32, max_value=64)
    )
    @settings(max_examples=10, deadline=15000)
    def test_throughput_scaling_properties(self, batch_sizes, channels, size):
        """Test properties of throughput benchmarking."""
        input_shape = (channels, size, size)
        
        # Benchmark throughput for different batch sizes
        throughput_results = self.benchmarker.benchmark_throughput(batch_sizes, input_shape)
        
        # Property 1: All throughput values should be non-negative
        for batch_size, throughput in throughput_results.items():
            assert throughput >= 0, f"Throughput for batch size {batch_size} should be non-negative, got {throughput}"
        
        # Property 2: Should have results for all requested batch sizes
        assert len(throughput_results) == len(batch_sizes), \
            f"Should have throughput results for all batch sizes, got {len(throughput_results)} vs {len(batch_sizes)}"
        
        # Property 3: Throughput should generally increase with batch size (up to memory limits)
        valid_throughputs = [(bs, tp) for bs, tp in throughput_results.items() if tp > 0]
        
        if len(valid_throughputs) >= 2:
            # Sort by batch size
            valid_throughputs.sort(key=lambda x: x[0])
            
            # Check that throughput doesn't decrease dramatically with larger batch sizes
            for i in range(1, len(valid_throughputs)):
                prev_bs, prev_tp = valid_throughputs[i-1]
                curr_bs, curr_tp = valid_throughputs[i]
                
                # Allow some decrease due to memory pressure, but not more than 50%
                assert curr_tp >= prev_tp * 0.5, \
                    f"Throughput shouldn't decrease dramatically: batch {prev_bs} -> {curr_bs}, " \
                    f"throughput {prev_tp:.2f} -> {curr_tp:.2f}"
    
    def test_hardware_config_detection(self):
        """Test hardware configuration detection."""
        config = self.benchmarker.hardware_config
        
        # Property 1: Device type should match the benchmarker's device
        assert config.device_type == self.device.type, \
            f"Hardware config device type should match benchmarker device, got {config.device_type} vs {self.device.type}"
        
        # Property 2: Device name should be a non-empty string
        assert isinstance(config.device_name, str), "Device name should be a string"
        assert len(config.device_name) > 0, "Device name should not be empty"
        
        # Property 3: Memory total should be positive
        assert config.memory_total_gb > 0, f"Total memory should be positive, got {config.memory_total_gb}"
        
        # Property 4: For CPU, compute capability should be None
        if config.device_type == "cpu":
            assert config.compute_capability is None, \
                f"CPU device should have None compute capability, got {config.compute_capability}"
    
    @given(
        channels=st.integers(min_value=3, max_value=4),
        size=st.integers(min_value=32, max_value=64)
    )
    @settings(max_examples=10, deadline=8000)
    def test_complete_benchmark_properties(self, channels, size):
        """Test properties of complete benchmark execution."""
        input_tensor = torch.randn(1, channels, size, size)
        
        # Run complete benchmark
        benchmark_result = self.benchmarker.run_complete_benchmark(input_tensor)
        
        # Property 1: Result should be PerformanceBenchmark instance
        assert isinstance(benchmark_result, PerformanceBenchmark), \
            f"Result should be PerformanceBenchmark instance, got {type(benchmark_result)}"
        
        # Property 2: All timing metrics should be positive
        assert benchmark_result.inference_time_ms > 0, \
            f"Inference time should be positive, got {benchmark_result.inference_time_ms}"
        assert benchmark_result.throughput_images_per_second > 0, \
            f"Throughput should be positive, got {benchmark_result.throughput_images_per_second}"
        
        # Property 3: Memory usage should be non-negative
        assert benchmark_result.memory_usage_mb >= 0, \
            f"Memory usage should be non-negative, got {benchmark_result.memory_usage_mb}"
        
        # Property 4: Utilization metrics should be in valid range
        assert 0 <= benchmark_result.gpu_utilization <= 100, \
            f"GPU utilization should be 0-100%, got {benchmark_result.gpu_utilization}"
        assert 0 <= benchmark_result.cpu_utilization <= 100, \
            f"CPU utilization should be 0-100%, got {benchmark_result.cpu_utilization}"
        
        # Property 5: Hardware config should be consistent
        assert benchmark_result.hardware_config.device_type == self.device.type, \
            f"Hardware config should match device type"
        
        # Property 6: Throughput should be consistent with inference time
        expected_throughput = 1000.0 / benchmark_result.inference_time_ms
        throughput_ratio = benchmark_result.throughput_images_per_second / expected_throughput
        
        # Allow some tolerance for measurement differences
        assert 0.5 <= throughput_ratio <= 2.0, \
            f"Throughput should be consistent with inference time, got ratio {throughput_ratio:.3f}"
    
    def test_benchmark_reproducibility(self):
        """Test that benchmarks are reasonably reproducible."""
        input_tensor = torch.randn(1, 3, 64, 64)
        
        # Run benchmark multiple times
        results = []
        for _ in range(3):
            result = self.benchmarker.run_complete_benchmark(input_tensor)
            results.append(result)
        
        # Property 1: Results should be reasonably consistent
        inference_times = [r.inference_time_ms for r in results]
        mean_time = np.mean(inference_times)
        
        if mean_time > 0:
            for time_ms in inference_times:
                relative_diff = abs(time_ms - mean_time) / mean_time
                assert relative_diff < 0.5, \
                    f"Inference times should be reasonably consistent, got relative difference {relative_diff:.3f}"
        
        # Property 2: Hardware config should be identical across runs
        for i in range(1, len(results)):
            assert results[i].hardware_config.device_type == results[0].hardware_config.device_type, \
                "Hardware config should be consistent across benchmark runs"
            assert results[i].hardware_config.device_name == results[0].hardware_config.device_name, \
                "Device name should be consistent across benchmark runs"


if __name__ == "__main__":
    pytest.main([__file__])