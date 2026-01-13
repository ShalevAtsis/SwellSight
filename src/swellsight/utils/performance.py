"""
Performance optimization utilities for real-time wave analysis.

Provides inference optimization, batch processing, and streaming capabilities
to achieve <200ms per image processing time.
"""

import time
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from contextlib import contextmanager

from .hardware import HardwareManager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for inference operations."""
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    throughput_fps: float
    memory_usage_mb: float
    gpu_utilization: float

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    target_latency_ms: float = 200.0
    enable_mixed_precision: bool = True
    enable_torch_compile: bool = True
    enable_tensorrt: bool = False
    batch_size: int = 1
    num_warmup_runs: int = 5
    enable_streaming: bool = False
    max_queue_size: int = 10

class PerformanceOptimizer:
    """Performance optimization manager for real-time inference."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize performance optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.hardware_manager = HardwareManager()
        self.device = self.hardware_manager.get_device()
        self._warmup_completed = False
        self._performance_history = []
        
        logger.info(f"Initialized PerformanceOptimizer with target latency: {self.config.target_latency_ms}ms")
    
    @contextmanager
    def measure_time(self, operation_name: str = "operation"):
        """Context manager for measuring execution time.
        
        Args:
            operation_name: Name of the operation being measured
            
        Yields:
            Dictionary to store timing results
        """
        timing_results = {}
        
        # GPU synchronization for accurate timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        try:
            yield timing_results
        finally:
            # GPU synchronization for accurate timing
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            timing_results[f"{operation_name}_time_ms"] = elapsed_ms
            
            logger.debug(f"{operation_name} took {elapsed_ms:.2f}ms")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply optimization techniques to model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        logger.info("Applying model optimizations...")
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        # Enable mixed precision if supported
        if self.config.enable_mixed_precision and self.device.type == "cuda":
            logger.info("Enabling mixed precision (FP16)")
            model = model.half()
        
        # Apply torch.compile if available (PyTorch 2.0+)
        if self.config.enable_torch_compile and hasattr(torch, 'compile'):
            try:
                logger.info("Applying torch.compile optimization")
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
                # Continue without torch.compile
                pass
        
        # Set inference mode optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return model
    
    def warmup_model(self, model: nn.Module, input_shape: Tuple[int, ...]) -> None:
        """Warm up model with dummy inputs for optimal performance.
        
        Args:
            model: Model to warm up
            input_shape: Shape of input tensor (without batch dimension)
        """
        if self._warmup_completed:
            return
        
        logger.info(f"Warming up model with {self.config.num_warmup_runs} runs...")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape, device=self.device)
        
        if self.config.enable_mixed_precision and self.device.type == "cuda":
            dummy_input = dummy_input.half()
        
        # Warmup runs
        with torch.no_grad():
            for i in range(self.config.num_warmup_runs):
                _ = model(dummy_input)
                
                # GPU synchronization
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        
        self._warmup_completed = True
        logger.info("Model warmup completed")
    
    def optimize_inference(self, model: nn.Module, input_tensor: torch.Tensor) -> Tuple[Any, PerformanceMetrics]:
        """Perform optimized inference with performance monitoring.
        
        Args:
            model: Optimized model
            input_tensor: Input tensor
            
        Returns:
            Tuple of (model output, performance metrics)
        """
        total_timing = {}
        
        with self.measure_time("total") as total_timing:
            # Preprocessing
            with self.measure_time("preprocessing") as prep_timing:
                # Move to device and apply precision
                input_tensor = input_tensor.to(self.device)
                
                if self.config.enable_mixed_precision and self.device.type == "cuda":
                    input_tensor = input_tensor.half()
            
            # Inference
            with self.measure_time("inference") as inf_timing:
                with torch.no_grad():
                    output = model(input_tensor)
            
            # Postprocessing
            with self.measure_time("postprocessing") as post_timing:
                # Move output to CPU if needed for further processing
                if isinstance(output, dict):
                    output = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                             for k, v in output.items()}
                elif isinstance(output, torch.Tensor):
                    output = output.cpu()
        
        # Calculate performance metrics
        metrics = PerformanceMetrics(
            inference_time_ms=inf_timing.get("inference_time_ms", 0.0),
            preprocessing_time_ms=prep_timing.get("preprocessing_time_ms", 0.0),
            postprocessing_time_ms=post_timing.get("postprocessing_time_ms", 0.0),
            total_time_ms=total_timing.get("total_time_ms", 0.0),
            throughput_fps=1000.0 / total_timing.get("total_time_ms", 1.0),
            memory_usage_mb=self._get_memory_usage(),
            gpu_utilization=self._get_gpu_utilization()
        )
        
        # Store performance history
        self._performance_history.append(metrics)
        
        # Log performance if target not met
        if metrics.total_time_ms > self.config.target_latency_ms:
            logger.warning(f"Target latency exceeded: {metrics.total_time_ms:.2f}ms > {self.config.target_latency_ms}ms")
        
        return output, metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            import psutil
            return psutil.Process().memory_info().rss / (1024 ** 2)
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if self.device.type == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(util.gpu)
            except:
                return 0.0
        return 0.0
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics from history.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self._performance_history:
            return {}
        
        total_times = [m.total_time_ms for m in self._performance_history]
        inference_times = [m.inference_time_ms for m in self._performance_history]
        throughputs = [m.throughput_fps for m in self._performance_history]
        
        return {
            "avg_total_time_ms": np.mean(total_times),
            "min_total_time_ms": np.min(total_times),
            "max_total_time_ms": np.max(total_times),
            "std_total_time_ms": np.std(total_times),
            "avg_inference_time_ms": np.mean(inference_times),
            "avg_throughput_fps": np.mean(throughputs),
            "target_met_percentage": np.mean([t <= self.config.target_latency_ms for t in total_times]) * 100,
            "num_samples": len(self._performance_history)
        }
    
    def clear_performance_history(self):
        """Clear performance history."""
        self._performance_history.clear()
        logger.info("Performance history cleared")

class StreamingProcessor:
    """Streaming processor for real-time wave analysis."""
    
    def __init__(self, model: nn.Module, optimizer: PerformanceOptimizer):
        """Initialize streaming processor.
        
        Args:
            model: Optimized model for inference
            optimizer: Performance optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.input_queue = queue.Queue(maxsize=optimizer.config.max_queue_size)
        self.output_queue = queue.Queue(maxsize=optimizer.config.max_queue_size)
        self.processing_thread = None
        self.is_running = False
        
        logger.info("Initialized StreamingProcessor")
    
    def start_streaming(self):
        """Start streaming processing thread."""
        if self.is_running:
            logger.warning("Streaming already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Streaming processing started")
    
    def stop_streaming(self):
        """Stop streaming processing."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Streaming processing stopped")
    
    def process_frame(self, input_tensor: torch.Tensor, timeout: float = 1.0) -> Optional[Tuple[Any, PerformanceMetrics]]:
        """Process a single frame through streaming pipeline.
        
        Args:
            input_tensor: Input tensor to process
            timeout: Timeout for getting result
            
        Returns:
            Tuple of (output, metrics) or None if timeout
        """
        try:
            # Add to input queue (non-blocking)
            self.input_queue.put_nowait(input_tensor)
            
            # Get result from output queue
            return self.output_queue.get(timeout=timeout)
            
        except queue.Full:
            logger.warning("Input queue full, dropping frame")
            return None
        except queue.Empty:
            logger.warning("Output queue empty, timeout reached")
            return None
    
    def _processing_loop(self):
        """Main processing loop for streaming."""
        logger.info("Starting streaming processing loop")
        
        while self.is_running:
            try:
                # Get input from queue
                input_tensor = self.input_queue.get(timeout=0.1)
                
                # Process with optimization
                output, metrics = self.optimizer.optimize_inference(self.model, input_tensor)
                
                # Put result in output queue (non-blocking)
                try:
                    self.output_queue.put_nowait((output, metrics))
                except queue.Full:
                    logger.warning("Output queue full, dropping result")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in streaming processing: {e}")
        
        logger.info("Streaming processing loop ended")

class BatchProcessor:
    """Batch processor for throughput optimization."""
    
    def __init__(self, model: nn.Module, optimizer: PerformanceOptimizer):
        """Initialize batch processor.
        
        Args:
            model: Optimized model for inference
            optimizer: Performance optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.optimal_batch_size = self._find_optimal_batch_size()
        
        logger.info(f"Initialized BatchProcessor with optimal batch size: {self.optimal_batch_size}")
    
    def _find_optimal_batch_size(self) -> int:
        """Find optimal batch size for current hardware."""
        # Use hardware manager to get optimal batch size
        input_size_mb = (4 * 518 * 518 * 4) / (1024 * 1024)  # 4 channels, 518x518, 4 bytes per float
        model_memory_mb = 3000  # Estimated model memory
        
        return self.optimizer.hardware_manager.get_optimal_batch_size(model_memory_mb, input_size_mb)
    
    def process_batch(self, input_tensors: List[torch.Tensor]) -> Tuple[List[Any], List[PerformanceMetrics]]:
        """Process a batch of inputs for optimal throughput.
        
        Args:
            input_tensors: List of input tensors
            
        Returns:
            Tuple of (outputs list, metrics list)
        """
        batch_size = len(input_tensors)
        
        if batch_size == 0:
            return [], []
        
        # Process in optimal batch sizes
        outputs = []
        metrics = []
        
        for i in range(0, batch_size, self.optimal_batch_size):
            batch_end = min(i + self.optimal_batch_size, batch_size)
            batch_inputs = input_tensors[i:batch_end]
            
            # Stack inputs into batch tensor
            batch_tensor = torch.stack(batch_inputs, dim=0)
            
            # Process batch
            batch_output, batch_metrics = self.optimizer.optimize_inference(self.model, batch_tensor)
            
            # Split outputs back to individual results
            if isinstance(batch_output, dict):
                for j in range(len(batch_inputs)):
                    single_output = {k: v[j] if isinstance(v, torch.Tensor) and v.dim() > 0 else v 
                                   for k, v in batch_output.items()}
                    outputs.append(single_output)
            else:
                for j in range(len(batch_inputs)):
                    outputs.append(batch_output[j])
            
            # Create individual metrics (approximate)
            for j in range(len(batch_inputs)):
                individual_metrics = PerformanceMetrics(
                    inference_time_ms=batch_metrics.inference_time_ms / len(batch_inputs),
                    preprocessing_time_ms=batch_metrics.preprocessing_time_ms / len(batch_inputs),
                    postprocessing_time_ms=batch_metrics.postprocessing_time_ms / len(batch_inputs),
                    total_time_ms=batch_metrics.total_time_ms / len(batch_inputs),
                    throughput_fps=batch_metrics.throughput_fps * len(batch_inputs),
                    memory_usage_mb=batch_metrics.memory_usage_mb,
                    gpu_utilization=batch_metrics.gpu_utilization
                )
                metrics.append(individual_metrics)
        
        return outputs, metrics