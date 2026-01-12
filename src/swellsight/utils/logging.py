"""
Logging setup and configuration for SwellSight system.

Provides structured logging with appropriate levels and formatting
for all system components.
"""

import logging
import logging.config
from typing import Optional
from pathlib import Path
import sys

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> None:
    """Setup logging configuration for SwellSight system.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional log file path
        log_format: Optional custom log format
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout
            }
        },
        "loggers": {
            "swellsight": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False
            },
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console"]
            }
        }
    }
    
    # Add file handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": log_level,
            "formatter": "standard",
            "filename": str(log_path),
            "mode": "a"
        }
        
        # Add file handler to loggers
        config["loggers"]["swellsight"]["handlers"].append("file")
        config["loggers"][""]["handlers"].append("file")
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Log setup completion
    logger = logging.getLogger("swellsight.logging")
    logger.info(f"Logging setup complete - Level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """Get logger instance for specified module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"swellsight.{name}")

class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__)