"""
Configuration Management for SwellSight Pipeline
Handles loading, validation, and management of pipeline configuration
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and validation for the SwellSight pipeline"""
    
    DEFAULT_CONFIG = {
        "pipeline": {
            "name": "swellsight_pipeline",
            "version": "1.0",
            "created": "2024-01-01T00:00:00Z"
        },
        "processing": {
            "batch_size": "auto",
            "max_images": 1000,
            "quality_threshold": 0.7,
            "memory_limit_gb": "auto"
        },
        "models": {
            "depth_model": "depth-anything/Depth-Anything-V2-Large",
            "base_model": "black-forest-labs/FLUX.1-dev",
            "controlnet_model": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
            "mixed_precision": True
        },
        "paths": {
            "data_dir": "./data",
            "output_dir": "./outputs",
            "checkpoint_dir": "./checkpoints"
        }
    }
    
    CONFIG_SCHEMA = {
        "type": "object",
        "required": ["pipeline", "processing", "models", "paths"],
        "properties": {
            "pipeline": {
                "type": "object",
                "required": ["name", "version"],
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "created": {"type": "string"}
                }
            },
            "processing": {
                "type": "object",
                "properties": {
                    "batch_size": {"oneOf": [{"type": "integer", "minimum": 1}, {"type": "string", "enum": ["auto"]}]},
                    "max_images": {"type": "integer", "minimum": 1},
                    "quality_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "memory_limit_gb": {"oneOf": [{"type": "number", "minimum": 0.1}, {"type": "string", "enum": ["auto"]}]}
                }
            },
            "models": {
                "type": "object",
                "properties": {
                    "depth_model": {"type": "string"},
                    "base_model": {"type": "string"},
                    "controlnet_model": {"type": "string"},
                    "mixed_precision": {"type": "boolean"}
                }
            },
            "paths": {
                "type": "object",
                "properties": {
                    "data_dir": {"type": "string"},
                    "output_dir": {"type": "string"},
                    "checkpoint_dir": {"type": "string"}
                }
            }
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file with fallback to defaults"""
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
            
            # Validate configuration
            if self.validate_config(merged_config):
                self.config = merged_config
                return self.config
            else:
                logger.warning("Configuration validation failed, using defaults")
                self.config = self.DEFAULT_CONFIG.copy()
                return self.config
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Falling back to default configuration")
            self.config = self.DEFAULT_CONFIG.copy()
            return self.config
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults to ensure all required keys exist"""
        merged = self.DEFAULT_CONFIG.copy()
        
        for section, values in config.items():
            if section in merged and isinstance(values, dict):
                merged[section].update(values)
            else:
                merged[section] = values
                
        return merged
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        try:
            # Basic structure validation
            required_sections = ["pipeline", "processing", "models", "paths"]
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate processing parameters
            processing = config.get("processing", {})
            
            # Validate batch_size
            batch_size = processing.get("batch_size")
            if batch_size != "auto" and (not isinstance(batch_size, int) or batch_size < 1):
                logger.warning(f"Invalid batch_size: {batch_size}, should be 'auto' or positive integer")
                return False
            
            # Validate quality_threshold
            quality_threshold = processing.get("quality_threshold", 0.7)
            if not isinstance(quality_threshold, (int, float)) or not (0.0 <= quality_threshold <= 1.0):
                logger.warning(f"Invalid quality_threshold: {quality_threshold}, should be between 0.0 and 1.0")
                return False
            
            # Validate memory_limit_gb
            memory_limit = processing.get("memory_limit_gb")
            if memory_limit != "auto" and (not isinstance(memory_limit, (int, float)) or memory_limit < 0.1):
                logger.warning(f"Invalid memory_limit_gb: {memory_limit}, should be 'auto' or >= 0.1")
                return False
            
            # Validate paths exist or can be created
            paths = config.get("paths", {})
            for path_name, path_value in paths.items():
                if not isinstance(path_value, str):
                    logger.warning(f"Invalid path {path_name}: {path_value}, should be string")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
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
        """Update configuration with new values"""
        try:
            if self.config is None:
                self.load_config()
            
            # Deep merge updates
            for section, values in updates.items():
                if section in self.config and isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
            
            # Validate updated configuration
            if self.validate_config(self.config):
                return True
            else:
                logger.error("Updated configuration failed validation")
                return False
                
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False


# Convenience functions for direct use in notebooks
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file with defaults"""
    manager = ConfigManager(config_path)
    return manager.load_config()

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration against schema"""
    manager = ConfigManager()
    return manager.validate_config(config)