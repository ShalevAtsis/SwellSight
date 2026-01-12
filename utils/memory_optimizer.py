"""
Memory Optimization Utilities for SwellSight Pipeline
Handles dynamic batch sizing, memory monitoring, and cleanup operations
"""

import gc
import psutil
import torch
import numpy as np
from typing import List, Any, Optional, Dict, Union
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Handles memory optimization and monitoring for pipeline operations"""
    
    def __init__(self, safety_margin: float = 0.1):
        """
        Initialize memory optimizer
        
        Args:
            safety_margin: Fraction of memory to keep as safety buffer (0.1 = 10%)
        """
        self.safety_margin = safety_margin
        self.gpu_available = torch.cuda.is_available()
        
    def get_optimal_batch_size(self, available_memory: Optional[int] = None, 
                             item_size: Optional[int] = None,
                             max_batch_size: int = 32) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            available_memory: Available memory in bytes (auto-detected if None)
            item_size: Memory per item in bytes (estimated if None)
            max_batch_size: Maximum allowed batch size
            
        Returns:
            Optimal batch size
        """
        try:
            # Get available memory
            if available_memory is None:
                if self.gpu_available:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_used = torch.cuda.memory_allocated(0)
                    available_memory = gpu_memory - gpu_used
                else:
                    # Use system RAM
                    memory_info = psutil.virtual_memory()
                    available_memory = memory_info.available
            
            # Apply safety margin
            usable_memory = int(available_memory * (1.0 - self.safety_margin))
            
            # Estimate item size if not provided
            if item_size is None:
                # Default estimate for typical image processing
                # Assume 512x512 RGB image with float32 processing
                item_size = 512 * 512 * 3 * 4  # ~3MB per image
                logger.info(f"Using estimated item size: {item_size / (1024*1024):.1f}MB")
            
            # Calculate optimal batch size
            if item_size > 0:
                optimal_batch = usable_memory // item_size
                optimal_batch = max(1, min(optimal_batch, max_batch_size))
            else:
                optimal_batch = max_batch_size
            
            logger.info(f"Calculated optimal batch size: {optimal_batch} "
                       f"(available memory: {available_memory / (1024*1024*1024):.1f}GB)")
            
            return optimal_batch
            
        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {e}")
            return min(4, max_batch_size)  # Conservative fallback
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            memory_info = {}
            
            # System memory
            system_memory = psutil.virtual_memory()
            memory_info['system_total_gb'] = system_memory.total / (1024**3)
            memory_info['system_used_gb'] = system_memory.used / (1024**3)
            memory_info['system_available_gb'] = system_memory.available / (1024**3)
            memory_info['system_percent'] = system_memory.percent
            
            # GPU memory if available
            if self.gpu_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_reserved = torch.cuda.memory_reserved(0)
                
                memory_info['gpu_total_gb'] = gpu_memory / (1024**3)
                memory_info['gpu_allocated_gb'] = gpu_allocated / (1024**3)
                memory_info['gpu_reserved_gb'] = gpu_reserved / (1024**3)
                memory_info['gpu_free_gb'] = (gpu_memory - gpu_reserved) / (1024**3)
                memory_info['gpu_percent'] = (gpu_allocated / gpu_memory) * 100
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error monitoring memory usage: {e}")
            return {}
    
    def cleanup_variables(self, variables: List[Any]) -> None:
        """
        Explicitly cleanup variables and call garbage collection
        
        Args:
            variables: List of variables to delete
        """
        try:
            for var in variables:
                if var is not None:
                    del var
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            if self.gpu_available:
                torch.cuda.empty_cache()
                
            logger.debug(f"Cleaned up {len(variables)} variables")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def suggest_memory_optimizations(self, current_usage: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Provide memory optimization suggestions based on current usage
        
        Args:
            current_usage: Current memory usage info (auto-detected if None)
            
        Returns:
            List of optimization suggestions
        """
        try:
            if current_usage is None:
                current_usage = self.monitor_memory_usage()
            
            suggestions = []
            
            # System memory suggestions
            if current_usage.get('system_percent', 0) > 80:
                suggestions.append("System memory usage is high (>80%). Consider reducing batch size.")
                suggestions.append("Close unnecessary applications to free up system memory.")
            
            # GPU memory suggestions
            if self.gpu_available and current_usage.get('gpu_percent', 0) > 80:
                suggestions.append("GPU memory usage is high (>80%). Consider reducing batch size.")
                suggestions.append("Use mixed precision training to reduce GPU memory usage.")
                suggestions.append("Clear GPU cache with torch.cuda.empty_cache().")
            
            # General suggestions
            if current_usage.get('system_percent', 0) > 60 or current_usage.get('gpu_percent', 0) > 60:
                suggestions.append("Process data in smaller batches to reduce memory pressure.")
                suggestions.append("Delete large variables when no longer needed.")
                suggestions.append("Use data generators instead of loading all data into memory.")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating memory optimization suggestions: {e}")
            return ["Unable to generate suggestions due to error."]
    
    def estimate_image_memory_usage(self, width: int = 512, height: int = 512, 
                                  channels: int = 3, dtype: str = 'float32') -> int:
        """
        Estimate memory usage for image processing
        
        Args:
            width: Image width
            height: Image height  
            channels: Number of channels
            dtype: Data type ('float32', 'uint8', etc.)
            
        Returns:
            Estimated memory usage in bytes
        """
        try:
            # Bytes per pixel based on data type
            dtype_sizes = {
                'uint8': 1,
                'float16': 2,
                'float32': 4,
                'float64': 8
            }
            
            bytes_per_pixel = dtype_sizes.get(dtype, 4)  # Default to float32
            
            # Calculate total memory
            # Multiply by 2 to account for intermediate processing steps
            memory_usage = width * height * channels * bytes_per_pixel * 2
            
            return memory_usage
            
        except Exception as e:
            logger.warning(f"Error estimating image memory usage: {e}")
            return 512 * 512 * 3 * 4 * 2  # Default estimate
    
    @contextmanager
    def memory_monitor(self, log_usage: bool = True):
        """
        Context manager for monitoring memory usage during operations
        
        Args:
            log_usage: Whether to log memory usage
        """
        try:
            # Record initial memory usage
            initial_usage = self.monitor_memory_usage()
            
            if log_usage:
                logger.info(f"Initial memory usage - "
                           f"System: {initial_usage.get('system_percent', 0):.1f}%, "
                           f"GPU: {initial_usage.get('gpu_percent', 0):.1f}%")
            
            # Yield control to the calling code
            yield self
            
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")
            raise
        finally:
            # Record final memory usage
            final_usage = self.monitor_memory_usage()
            
            if log_usage:
                logger.info(f"Final memory usage - "
                           f"System: {final_usage.get('system_percent', 0):.1f}%, "
                           f"GPU: {final_usage.get('gpu_percent', 0):.1f}%")
            
            # Provide suggestions if memory usage is high
            if (final_usage.get('system_percent', 0) > 80 or 
                final_usage.get('gpu_percent', 0) > 80):
                suggestions = self.suggest_memory_optimizations(final_usage)
                for suggestion in suggestions[:3]:  # Show top 3 suggestions
                    logger.warning(f"Memory optimization suggestion: {suggestion}")


# Convenience functions for direct use in notebooks
def get_optimal_batch_size(available_memory: Optional[int] = None, 
                          item_size: Optional[int] = None,
                          max_batch_size: int = 32) -> int:
    """Calculate optimal batch size based on available memory"""
    optimizer = MemoryOptimizer()
    return optimizer.get_optimal_batch_size(available_memory, item_size, max_batch_size)

def cleanup_variables(variables: List[Any]) -> None:
    """Cleanup variables and call garbage collection"""
    optimizer = MemoryOptimizer()
    optimizer.cleanup_variables(variables)

def monitor_memory() -> Dict[str, float]:
    """Monitor current memory usage"""
    optimizer = MemoryOptimizer()
    return optimizer.monitor_memory_usage()

def estimate_image_memory_usage(width: int = 512, height: int = 512, 
                               channels: int = 3, dtype: str = 'float32') -> int:
    """Estimate memory usage for image processing"""
    optimizer = MemoryOptimizer()
    return optimizer.estimate_image_memory_usage(width, height, channels, dtype)