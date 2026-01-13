"""
Performance benchmarking system for wave analysis models.

Implements inference speed benchmarking, memory usage profiling,
and throughput testing across different hardware configurations.
"""

from typing import Dict, Any, List
import time
import torch
import psutil
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class HardwareConfig:
    """Hardware configuration information."""
    device_type: str  # "cuda" or "cpu"
    device_name: str
    memory_total_gb: float
    compute_capability: str = None  # For CUDA devices

@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    inference_time_ms: float
    memory_usage_mb: float
    throughput_images_per_second: float
    gpu_utilization: float
    cpu_utilization: float
    hardware_config: HardwareConfig

class PerformanceBenchmarker:
    """Performance benchmarking system for wave analysis models."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """Initialize performance benchmarker.
        
        Args:
            model: Model to benchmark
            device: Device to run benchmarks on
        """
        self.model = model
        self.device = device
        self.hardware_config = self._detect_hardware_config()
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
    
    def _detect_hardware_config(self) -> HardwareConfig:
        """Detect current hardware configuration."""
        device_type = self.device.type
        
        if device_type == "cuda":
            device_name = torch.cuda.get_device_name(self.device)
            memory_total_gb = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            compute_capability = f"{torch.cuda.get_device_properties(self.device).major}.{torch.cuda.get_device_properties(self.device).minor}"
        else:
            device_name = "CPU"
            memory_total_gb = psutil.virtual_memory().total / (1024**3)
            compute_capability = None
        
        return HardwareConfig(
            device_type=device_type,
            device_name=device_name,
            memory_total_gb=memory_total_gb,
            compute_capability=compute_capability
        )
    
    @contextmanager
    def _measure_time(self):
        """Context manager for measuring execution time."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        yield
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        self.last_execution_time = (end_time - start_time) * 1000  # Convert to ms
    
    def benchmark_inference_speed(self, 
                                 input_tensor: torch.Tensor,
                                 num_warmup: int = 10,
                                 num_iterations: int = 100) -> float:
        """Benchmark model inference speed.
        
        Args:
            input_tensor: Sample input tensor
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
            
        Returns:
            Average inference time in milliseconds
        """
        input_tensor = input_tensor.to(self.device)
        
        # Warmup iterations
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(input_tensor)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
        # Benchmark iterations
        execution_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                with self._measure_time():
                    _ = self.model(input_tensor)
                execution_times.append(self.last_execution_time)
        
        return np.mean(execution_times)
    
    def measure_memory_usage(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Measure memory usage during inference.
        
        Args:
            input_tensor: Sample input tensor
            
        Returns:
            Dictionary with memory usage statistics
        """
        input_tensor = input_tensor.to(self.device)
        
        # Clear cache and measure baseline
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            baseline_memory = 0.0
        
        # Measure memory during inference
        with torch.no_grad():
            _ = self.model(input_tensor)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                gpu_utilization = (current_memory / self.hardware_config.memory_total_gb / 1024) * 100
            else:
                peak_memory = 0.0
                current_memory = 0.0
                gpu_utilization = 0.0
        
        return {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "current_memory_mb": current_memory,
            "memory_increase_mb": peak_memory - baseline_memory,
            "gpu_utilization": gpu_utilization
        }
    
    def benchmark_throughput(self, 
                           batch_sizes: List[int],
                           input_shape: tuple) -> Dict[int, float]:
        """Benchmark throughput for different batch sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            input_shape: Shape of individual input (C, H, W)
            
        Returns:
            Dictionary mapping batch size to throughput (images/second)
        """
        throughput_results = {}
        
        for batch_size in batch_sizes:
            try:
                # Create batch input tensor
                batch_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Benchmark this batch size
                avg_time_ms = self.benchmark_inference_speed(
                    batch_input, num_warmup=5, num_iterations=20
                )
                
                # Calculate throughput (images per second)
                throughput = (batch_size * 1000.0) / avg_time_ms
                throughput_results[batch_size] = throughput
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Skip this batch size if OOM
                    throughput_results[batch_size] = 0.0
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise e
        
        return throughput_results
    
    def run_complete_benchmark(self, 
                             input_tensor: torch.Tensor) -> PerformanceBenchmark:
        """Run complete performance benchmark.
        
        Args:
            input_tensor: Sample input tensor for benchmarking
            
        Returns:
            PerformanceBenchmark with comprehensive performance metrics
        """
        # Benchmark inference speed
        inference_time = self.benchmark_inference_speed(input_tensor)
        
        # Measure memory usage
        memory_stats = self.measure_memory_usage(input_tensor)
        
        # Calculate throughput
        throughput = 1000.0 / inference_time  # images per second
        
        return PerformanceBenchmark(
            inference_time_ms=inference_time,
            memory_usage_mb=memory_stats.get("peak_memory_mb", 0),
            throughput_images_per_second=throughput,
            gpu_utilization=memory_stats.get("gpu_utilization", 0),
            cpu_utilization=psutil.cpu_percent(),
            hardware_config=self.hardware_config
        )