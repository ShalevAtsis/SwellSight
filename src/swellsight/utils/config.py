"""
Configuration management system for model parameters and settings.

Provides centralized configuration loading, validation, and management
for all SwellSight components.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
import json
from dataclasses import dataclass, field
import logging

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    depth_model_size: str = "large"
    depth_precision: str = "fp16"
    backbone_model: str = "dinov2-base"
    freeze_backbone: bool = True
    input_resolution: tuple = (518, 518)
    num_classes_direction: int = 3
    num_classes_breaking: int = 3

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = True
    save_checkpoint_every: int = 10
    validate_every: int = 5
    early_stopping_patience: int = 20

@dataclass
class DataConfig:
    """Configuration for data processing."""
    min_resolution: tuple = (640, 480)
    max_resolution: tuple = (3840, 2160)
    quality_threshold: float = 0.5
    augmentation_enabled: bool = True
    synthetic_data_ratio: float = 0.7

@dataclass
class SystemConfig:
    """Configuration for system settings."""
    use_gpu: bool = True
    max_processing_time: float = 30.0
    confidence_threshold: float = 0.7
    log_level: str = "INFO"
    output_dir: str = "outputs"

@dataclass
class SwellSightConfig:
    """Complete SwellSight configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

class ConfigManager:
    """Configuration manager for SwellSight system."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[SwellSightConfig] = None
        self.logger = logging.getLogger(__name__)
        
        if self.config_path and self.config_path.exists():
            self.load_config()
        else:
            self.config = SwellSightConfig()  # Use defaults
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> SwellSightConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded SwellSightConfig instance
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            self.logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            self.config = SwellSightConfig()
            return self.config
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            # Convert dictionary to config objects
            self.config = self._dict_to_config(config_dict)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            self.config = SwellSightConfig()  # Fallback to defaults
        
        return self.config
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path:
            raise ValueError("No config path specified")
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = self._config_to_dict(self.config)
        
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif self.config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {self.config_path}: {e}")
            raise
    
    def validate_config(self) -> bool:
        """Validate current configuration.
        
        Returns:
            True if configuration is valid
        """
        if not self.config:
            return False
        
        try:
            # Validate model config
            assert self.config.model.depth_model_size in ["small", "base", "large"]
            assert self.config.model.depth_precision in ["fp16", "fp32"]
            assert len(self.config.model.input_resolution) == 2
            
            # Validate training config
            assert self.config.training.batch_size > 0
            assert self.config.training.learning_rate > 0
            assert self.config.training.num_epochs > 0
            
            # Validate data config
            assert len(self.config.data.min_resolution) == 2
            assert len(self.config.data.max_resolution) == 2
            assert 0 <= self.config.data.quality_threshold <= 1
            
            # Validate system config
            assert self.config.system.max_processing_time > 0
            assert 0 <= self.config.system.confidence_threshold <= 1
            
            return True
            
        except AssertionError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config(self) -> SwellSightConfig:
        """Get current configuration.
        
        Returns:
            Current SwellSightConfig instance
        """
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
        """
        # TODO: Implement configuration updates
        self.logger.info("Configuration updated")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SwellSightConfig:
        """Convert dictionary to configuration objects."""
        # TODO: Implement dictionary to config conversion
        return SwellSightConfig()
    
    def _config_to_dict(self, config: SwellSightConfig) -> Dict[str, Any]:
        """Convert configuration objects to dictionary."""
        # TODO: Implement config to dictionary conversion
        return {}

def load_config(config_path: Union[str, Path]) -> SwellSightConfig:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded SwellSightConfig instance
    """
    manager = ConfigManager(config_path)
    return manager.get_config()