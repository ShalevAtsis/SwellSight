#!/usr/bin/env python3
"""
Training script for SwellSight Wave Analysis System.

Provides command-line interface for training wave analysis models
with configurable parameters and monitoring.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.utils.config import ConfigManager
from swellsight.utils.logging import setup_logging
from swellsight.training.trainer import WaveAnalysisTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SwellSight Wave Analysis System"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to training data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training",
        help="Path to output directory for checkpoints and logs"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to use (default: auto-detect)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level)
    logger = logging.getLogger("swellsight.train")
    
    logger.info("Starting SwellSight training...")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.get_config()
        
        # Validate configuration
        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            return 1
        
        # TODO: Implement training pipeline in task 7.1
        logger.info("Training pipeline will be implemented in task 7.1")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())