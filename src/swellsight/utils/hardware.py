"""
Hardware detection and management for GPU/CPU optimization.

Provides hardware detection, GPU memory management, and automatic
fallback mechanisms for optimal performance.
"""

import torch
import psutil
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class HardwareInfo:
    """Hardware information and capabilities."""
    device_type: str  # "cuda" or "cpu"
    device_name: str
    device_count: int
    memory_total_gb: float
    memory_available_gb: float
    compute_capability: Optional[str] = None
    cuda_version: Optional[str] = None

class HardwareManager:
    """Hardware detection and management system."""
    
    def __init__(self):
        """Initialize hardware manager."""
        self.logger = logging.getLogger(__name__)
        self.hardware_info = self.detect_hardware()
        self.preferred_device = self._select_preferred_device()
    
    def detect_hardware(self) -> HardwareInfo:
        """Detect available hardware capabilities.
        
        Returns:
            HardwareInfo with detected capabilities
        """
        # Check CUDA availability
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            
            # Get memory info
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_total_gb = memory_total / (1024**3)
            
            # Get available memory
            torch.cuda.empty_cache()
            memory_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            memory_available_gb = memory_free / (1024**3)
            
            # Get compute capability
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
            
            return HardwareInfo(
                device_type="cuda",
                device_name=device_name,
                device_count=device_count,
                memory_total_gb=memory_total_gb,
                memory_available_gb=memory_available_gb,
                compute_capability=compute_capability,
                cuda_version=torch.version.cuda
            )
        else:
            # CPU-only system
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            memory_total_gb = memory_info.total / (1024**3)
            memory_available_gb = memory_info.available / (1024**3)
            
            return HardwareInfo(
                device_type="cpu",
                device_name="CPU",
                device_count=cpu_count,
                memory_total_gb=memory_total_gb,
                memory_available_gb=memory_available_gb
            )
    
    def _select_preferred_device(self) -> torch.device:
        """Select preferred device based on hardware capabilities."""
        if self.hardware_info.device_type == "cuda":
            # Check if we have sufficient GPU memory (minimum 4GB for inference)
            if self.hardware_info.memory_available_gb >= 4.0:
                device = torch.device("cuda:0")
                self.logger.info(f"Selected GPU device: {self.hardware_info.device_name}")
            else:
                device = torch.device("cpu")
                self.logger.warning(f"Insufficient GPU memory ({self.hardware_info.memory_available_gb:.1f}GB), falling back to CPU")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU device")
        
        return device
    
    def get_optimal_batch_size(self, 
                              model_memory_mb: float,
                              input_size_mb: float) -> int:
        """Calculate optimal batch size based on available memory.
        
        Args:
            model_memory_mb: Model memory usage in MB
            input_size_mb: Single input memory usage in MB
            
        Returns:
            Optimal batch size for current hardware
        """
        if self.hardware_info.device_type == "cuda":
            # Reserve 20% of GPU memory for other operations
            available_memory_mb = self.hardware_info.memory_available_gb * 1024 * 0.8
            
            # Calculate batch size
            memory_per_sample = input_size_mb * 2  # Forward + backward pass
            usable_memory = available_memory_mb - model_memory_mb
            
            if usable_memory <= 0:
                self.logger.warning("Insufficient memory for model, using batch size 1")
                return 1
            
            batch_size = max(1, int(usable_memory / memory_per_sample))
            
        else:
            # CPU: Use smaller batch sizes to avoid memory issues
            available_memory_mb = self.hardware_info.memory_available_gb * 1024 * 0.5
            memory_per_sample = input_size_mb
            usable_memory = available_memory_mb - model_memory_mb
            
            batch_size = max(1, min(16, int(usable_memory / memory_per_sample)))
        
        self.logger.info(f"Optimal batch size: {batch_size}")
        return batch_size
    
    def check_memory_requirements(self, required_memory_gb: float) -> bool:
        """Check if system has sufficient memory for operation.
        
        Args:
            required_memory_gb: Required memory in GB
            
        Returns:
            True if sufficient memory is available
        """
        available = self.hardware_info.memory_available_gb
        if available >= required_memory_gb:
            return True
        else:
            self.logger.warning(f"Insufficient memory: {available:.1f}GB available, {required_memory_gb:.1f}GB required")
            return False
    
    def get_device(self, force_cpu: bool = False) -> torch.device:
        """Get appropriate device for computation.
        
        Args:
            force_cpu: Force CPU usage even if GPU is available
            
        Returns:
            PyTorch device for computation
        """
        if force_cpu:
            return torch.device("cpu")
        return self.preferred_device
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cache cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information.
        
        Returns:
            Dictionary with system information
        """
        return {
            "hardware_info": self.hardware_info,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "preferred_device": str(self.preferred_device),
            "cpu_count": psutil.cpu_count(),
            "system_memory_gb": psutil.virtual_memory().total / (1024**3)
        }