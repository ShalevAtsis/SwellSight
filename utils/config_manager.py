"""
Enhanced Configuration Management for SwellSight Pipeline
Handles loading, validation, hardware detection, and management of pipeline configuration
"""

import json
import os
import platform
import psutil
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConfigManager:
    """Enhanced configuration manager with hardware detection and adaptive settings"""
    
    DEFAULT_CONFIG = {
        "pipeline": {
            "name": "swellsight_pipeline",
            "version": "2.0",
            "created": "2024-01-12T00:00:00Z",
            "description": "Enhanced SwellSight pipeline with Depth-Anything-V2 and FLUX.1-dev support"
        },
        "processing": {
            "batch_size": "auto",
            "max_images": 1000,
            "quality_threshold": 0.7,
            "memory_limit_gb": "auto",
            "image_resolution": {
                "default": 512,
                "max": 1024,
                "min": 256
            },
            "parallel_workers": "auto"
        },
        "models": {
            "depth_model": {
                "name": "depth-anything/Depth-Anything-V2-Large",
                "type": "depth_estimation",
                "input_size": [518, 518],
                "output_channels": 1,
                "precision": "fp16"
            },
            "base_model": {
                "name": "black-forest-labs/FLUX.1-dev",
                "type": "text_to_image",
                "input_size": [1024, 1024],
                "guidance_scale": 3.5,
                "num_inference_steps": 28,
                "precision": "fp16"
            },
            "controlnet_model": {
                "name": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
                "type": "controlnet",
                "conditioning_scale": 0.6,
                "control_guidance_start": 0.0,
                "control_guidance_end": 1.0,
                "precision": "fp16"
            },
            "mixed_precision": True,
            "enable_xformers": True,
            "enable_flash_attention": False,
            "gradient_checkpointing": False,
            "cpu_offload": False
        },
        "paths": {
            "data_dir": "./data",
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints",
            "models_cache_dir": "./models",
            "logs_dir": "./logs"
        },
        "hardware": {
            "auto_detect": True,
            "gpu_memory_fraction": 0.9,
            "cpu_threads": "auto",
            "memory_growth": True
        },
        "optimizations": {
            "use_xformers": "auto",
            "use_flash_attention": "auto",
            "gradient_checkpointing": "auto",
            "cpu_offload": "auto",
            "compile_models": False,
            "cache_models": True
        },
        "validation": {
            "validate_inputs": True,
            "quality_checks": True,
            "memory_monitoring": True,
            "progress_tracking": True
        },
        "logging": {
            "level": "INFO",
            "save_logs": True,
            "log_file": "pipeline.log"
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = None
        self.hardware_info = None
        
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect hardware capabilities and return hardware information"""
        hardware_info = {
            'platform': platform.system(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_memory_gb': 0,
            'gpu_name': None,
            'cuda_version': None,
            'mixed_precision_supported': False,
            'flux_compatibility': 'none'
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            hardware_info.update({
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'gpu_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda
            })
            
            # Check mixed precision support (Tensor Cores)
            gpu_capability = torch.cuda.get_device_capability(0)
            hardware_info['mixed_precision_supported'] = gpu_capability[0] >= 7
            
            # Assess FLUX compatibility based on GPU memory
            gpu_memory = hardware_info['gpu_memory_gb']
            if gpu_memory >= 24:
                hardware_info['flux_compatibility'] = 'optimal'
            elif gpu_memory >= 16:
                hardware_info['flux_compatibility'] = 'adequate'
            elif gpu_memory >= 8:
                hardware_info['flux_compatibility'] = 'minimal'
            else:
                hardware_info['flux_compatibility'] = 'insufficient'
        
        self.hardware_info = hardware_info
        return hardware_info
    
    def adapt_config_to_hardware(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt configuration based on detected hardware capabilities"""
        if not self.hardware_info:
            self.detect_hardware()
        
        adapted_config = config.copy()
        
        # Adapt batch size based on GPU memory
        if config['processing']['batch_size'] == 'auto':
            if self.hardware_info['flux_compatibility'] == 'optimal':
                adapted_config['processing']['batch_size'] = 8
            elif self.hardware_info['flux_compatibility'] == 'adequate':
                adapted_config['processing']['batch_size'] = 4
            elif self.hardware_info['flux_compatibility'] == 'minimal':
                adapted_config['processing']['batch_size'] = 2
            else:
                adapted_config['processing']['batch_size'] = 1
        
        # Adapt parallel workers based on CPU count
        if config['processing']['parallel_workers'] == 'auto':
            adapted_config['processing']['parallel_workers'] = min(self.hardware_info['cpu_count'], 8)
        
        # Adapt memory limit based on available memory
        if config['processing']['memory_limit_gb'] == 'auto':
            adapted_config['processing']['memory_limit_gb'] = self.hardware_info['memory_gb'] * 0.8
        
        # Adapt CPU threads
        if config['hardware']['cpu_threads'] == 'auto':
            adapted_config['hardware']['cpu_threads'] = self.hardware_info['cpu_count']
        
        # Adapt optimizations based on hardware
        if not self.hardware_info['gpu_available']:
            # CPU-only optimizations
            adapted_config['models']['mixed_precision'] = False
            adapted_config['models']['cpu_offload'] = True
            adapted_config['optimizations']['use_xformers'] = False
            adapted_config['optimizations']['use_flash_attention'] = False
            adapted_config['optimizations']['gradient_checkpointing'] = True
        else:
            # GPU optimizations based on capability
            if not self.hardware_info['mixed_precision_supported']:
                adapted_config['models']['mixed_precision'] = False
            
            if self.hardware_info['flux_compatibility'] in ['minimal', 'insufficient']:
                adapted_config['models']['gradient_checkpointing'] = True
                adapted_config['optimizations']['gradient_checkpointing'] = True
                
            if self.hardware_info['flux_compatibility'] == 'insufficient':
                adapted_config['models']['cpu_offload'] = True
                adapted_config['optimizations']['cpu_offload'] = True
        
        # Store hardware info in config
        adapted_config['detected_hardware'] = self.hardware_info
        
        return adapted_config
    
    def validate_parameter_ranges(self, config: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate parameter ranges and provide warnings for unusual values"""
        warnings = []
        
        # Validate processing parameters
        processing = config.get('processing', {})
        
        # Check batch size
        batch_size = processing.get('batch_size', 1)
        if isinstance(batch_size, int):
            if batch_size > 16:
                warnings.append(f"Large batch size ({batch_size}) may cause memory issues")
            elif batch_size < 1:
                warnings.append(f"Invalid batch size ({batch_size}), must be >= 1")
        
        # Check quality threshold
        quality_threshold = processing.get('quality_threshold', 0.7)
        if quality_threshold < 0.3:
            warnings.append(f"Very low quality threshold ({quality_threshold}) may include poor quality data")
        elif quality_threshold > 0.95:
            warnings.append(f"Very high quality threshold ({quality_threshold}) may exclude too much data")
        
        # Check image resolution
        img_res = processing.get('image_resolution', {})
        if isinstance(img_res, dict):
            default_res = img_res.get('default', 512)
            if default_res > 1024:
                warnings.append(f"High default resolution ({default_res}) will require significant GPU memory")
            elif default_res < 256:
                warnings.append(f"Low default resolution ({default_res}) may reduce quality")
        
        # Validate model parameters
        models = config.get('models', {})
        
        # Check FLUX parameters
        base_model = models.get('base_model', {})
        if isinstance(base_model, dict):
            steps = base_model.get('num_inference_steps', 28)
            if steps > 50:
                warnings.append(f"High inference steps ({steps}) will increase generation time")
            elif steps < 10:
                warnings.append(f"Low inference steps ({steps}) may reduce quality")
            
            guidance = base_model.get('guidance_scale', 3.5)
            if guidance > 10:
                warnings.append(f"High guidance scale ({guidance}) may cause over-saturation")
            elif guidance < 1:
                warnings.append(f"Low guidance scale ({guidance}) may ignore prompts")
        
        return len(warnings) == 0, warnings
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file with hardware adaptation and validation"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                config = self.DEFAULT_CONFIG.copy()
                logger.info("Using default configuration (config file not found)")
                
            # Merge with defaults to ensure all keys exist
            merged_config = self._merge_with_defaults(config)
            
            # Adapt configuration to hardware if auto_detect is enabled
            if merged_config.get('hardware', {}).get('auto_detect', True):
                merged_config = self.adapt_config_to_hardware(merged_config)
                logger.info("Configuration adapted to detected hardware")
            
            # Validate configuration and parameters
            is_valid, validation_errors = self.validate_config(merged_config)
            param_valid, param_warnings = self.validate_parameter_ranges(merged_config)
            
            if is_valid:
                self.config = merged_config
                
                # Log parameter warnings
                if param_warnings:
                    logger.warning("Parameter validation warnings:")
                    for warning in param_warnings:
                        logger.warning(f"  - {warning}")
                
                return self.config
            else:
                logger.error(f"Configuration validation failed: {validation_errors}")
                logger.info("Falling back to default configuration")
                self.config = self.DEFAULT_CONFIG.copy()
                return self.config
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Falling back to default configuration")
            self.config = self.DEFAULT_CONFIG.copy()
            return self.config
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults"""
        def deep_merge(default: dict, user: dict) -> dict:
            result = default.copy()
            for key, value in user.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(self.DEFAULT_CONFIG, config)
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate configuration structure and return validation results"""
        errors = []
        
        try:
            # Check required top-level sections
            required_sections = ["pipeline", "processing", "models", "paths"]
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing required configuration section: {section}")
            
            # Validate pipeline section
            pipeline = config.get("pipeline", {})
            if not isinstance(pipeline.get("name"), str):
                errors.append("Pipeline name must be a string")
            if not isinstance(pipeline.get("version"), str):
                errors.append("Pipeline version must be a string")
            
            # Validate processing section
            processing = config.get("processing", {})
            batch_size = processing.get("batch_size")
            if batch_size != "auto" and (not isinstance(batch_size, int) or batch_size < 1):
                errors.append(f"Invalid batch_size: {batch_size}")
            
            quality_threshold = processing.get("quality_threshold", 0.7)
            if not isinstance(quality_threshold, (int, float)) or not (0.0 <= quality_threshold <= 1.0):
                errors.append(f"Invalid quality_threshold: {quality_threshold}")
            
            # Validate models section
            models = config.get("models", {})
            required_models = ["depth_model", "base_model", "controlnet_model"]
            for model in required_models:
                if model not in models:
                    errors.append(f"Missing required model: {model}")
            
            # Validate paths section
            paths = config.get("paths", {})
            required_paths = ["data_dir", "output_dir", "checkpoint_dir"]
            for path_name in required_paths:
                if path_name not in paths:
                    errors.append(f"Missing required path: {path_name}")
                elif not isinstance(paths[path_name], str):
                    errors.append(f"Path {path_name} must be a string")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
            return False, errors
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save current or provided configuration to file"""
        try:
            config_to_save = config or self.config
            if not config_to_save:
                logger.error("No configuration to save")
                return False
                
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration, loading if necessary"""
        if self.config is None:
            return self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values and validate"""
        try:
            if self.config is None:
                self.load_config()
            
            # Deep merge updates
            def deep_update(base: dict, updates: dict) -> dict:
                for key, value in updates.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_update(base[key], value)
                    else:
                        base[key] = value
                return base
            
            deep_update(self.config, updates)
            
            # Validate updated configuration
            is_valid, errors = self.validate_config(self.config)
            if is_valid:
                return True
            else:
                logger.error(f"Updated configuration failed validation: {errors}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get detected hardware information"""
        if self.hardware_info is None:
            self.detect_hardware()
        return self.hardware_info


# Enhanced convenience functions for direct use in notebooks
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file with hardware adaptation and defaults"""
    manager = ConfigManager(config_path)
    return manager.load_config()

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration and return validation results"""
    manager = ConfigManager()
    is_valid, errors = manager.validate_config(config)
    param_valid, warnings = manager.validate_parameter_ranges(config)
    
    return {
        'valid': is_valid and param_valid,
        'errors': errors,
        'warnings': warnings
    }

def detect_hardware() -> Dict[str, Any]:
    """Detect and return hardware information"""
    manager = ConfigManager()
    return manager.detect_hardware()

def adapt_config_to_hardware(config: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt configuration to detected hardware"""
    manager = ConfigManager()
    return manager.adapt_config_to_hardware(config)