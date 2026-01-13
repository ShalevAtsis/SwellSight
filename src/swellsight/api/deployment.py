"""
Deployment and model serving utilities for SwellSight API.

Provides utilities for model loading, initialization, health monitoring,
and graceful shutdown handling.
"""

import logging
import time
import os
import json
import psutil
from typing import Dict, Any, Optional
from pathlib import Path

from ..core.pipeline import WaveAnalysisPipeline
from ..utils.logging import setup_logging

logger = logging.getLogger("swellsight.deployment")

class ModelServer:
    """Model server for managing pipeline lifecycle and health monitoring."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize model server.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.pipeline: Optional[WaveAnalysisPipeline] = None
        self.config = self._load_config(config_path)
        self.start_time = time.time()
        self.health_status = {"status": "initializing"}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load deployment configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "model_cache_dir": "models",
            "max_memory_usage_gb": 8.0,
            "enable_gpu": True,
            "batch_size": 4,
            "cache_ttl_seconds": 300,
            "health_check_interval": 30
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def initialize_models(self) -> bool:
        """Initialize all models and pipeline components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing SwellSight pipeline...")
            self.health_status = {"status": "initializing", "stage": "pipeline"}
            
            # Create pipeline with configuration
            self.pipeline = WaveAnalysisPipeline()
            
            # Verify initialization
            status = self.pipeline.get_pipeline_status()
            if status.get("components_initialized", False):
                self.health_status = {
                    "status": "healthy",
                    "initialized_at": time.time(),
                    "pipeline_status": status
                }
                logger.info("Pipeline initialization successful")
                return True
            else:
                self.health_status = {
                    "status": "degraded",
                    "error": "Some components failed to initialize",
                    "pipeline_status": status
                }
                logger.warning("Pipeline initialization completed with warnings")
                return False
                
        except Exception as e:
            self.health_status = {
                "status": "error",
                "error": str(e),
                "failed_at": time.time()
            }
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status with system metrics.
        
        Returns:
            Health status dictionary
        """
        try:
            # Get system metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            health_info = {
                **self.health_status,
                "uptime_seconds": time.time() - self.start_time,
                "system_metrics": {
                    "memory_usage_percent": memory_info.percent,
                    "memory_available_gb": memory_info.available / (1024**3),
                    "cpu_usage_percent": cpu_percent
                },
                "timestamp": time.time()
            }
            
            # Add pipeline-specific metrics if available
            if self.pipeline:
                try:
                    pipeline_status = self.pipeline.get_pipeline_status()
                    health_info["pipeline_metrics"] = pipeline_status
                except Exception as e:
                    health_info["pipeline_error"] = str(e)
            
            return health_info
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Health check failed: {str(e)}",
                "timestamp": time.time()
            }
    
    def reload_models(self) -> Dict[str, Any]:
        """Reload all models without restarting the server.
        
        Returns:
            Reload operation status
        """
        try:
            logger.info("Starting model reload operation")
            old_status = self.health_status.copy()
            
            if self.pipeline:
                # Attempt to reinitialize pipeline
                try:
                    self.pipeline._initialize_components()
                    new_status = self.pipeline.get_pipeline_status()
                    
                    self.health_status = {
                        "status": "healthy" if new_status.get("components_initialized") else "degraded",
                        "reloaded_at": time.time(),
                        "pipeline_status": new_status
                    }
                    
                    return {
                        "status": "success",
                        "message": "Models reloaded successfully",
                        "old_status": old_status,
                        "new_status": self.health_status,
                        "timestamp": time.time()
                    }
                    
                except Exception as e:
                    self.health_status = {
                        "status": "error",
                        "error": f"Reload failed: {str(e)}",
                        "failed_at": time.time()
                    }
                    raise
            else:
                # Initialize pipeline if not exists
                success = self.initialize_models()
                return {
                    "status": "success" if success else "partial_failure",
                    "message": "Pipeline initialized" if success else "Pipeline initialization had issues",
                    "new_status": self.health_status,
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            return {
                "status": "error",
                "message": f"Model reload failed: {str(e)}",
                "timestamp": time.time()
            }
    
    def cleanup(self):
        """Clean up resources and prepare for shutdown."""
        try:
            logger.info("Starting cleanup process...")
            
            if self.pipeline:
                # Add any specific cleanup methods if available
                if hasattr(self.pipeline, 'cleanup'):
                    self.pipeline.cleanup()
                
                # Clear pipeline reference
                self.pipeline = None
            
            self.health_status = {"status": "shutdown", "shutdown_at": time.time()}
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def create_model_server(config_path: Optional[str] = None) -> ModelServer:
    """Create and initialize model server.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Initialized ModelServer instance
    """
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Create server
    server = ModelServer(config_path)
    
    # Initialize models
    success = server.initialize_models()
    if not success:
        logger.warning("Model server initialized with warnings")
    
    return server

if __name__ == "__main__":
    """Run model server initialization for testing."""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    server = create_model_server(config_path)
    
    print("Model server status:")
    print(json.dumps(server.get_health_status(), indent=2))