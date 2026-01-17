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
import torch

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
    log_file = Path(args.output_dir) / "training.log"
    setup_logging(log_level=log_level, log_file=str(log_file))
    logger = logging.getLogger("swellsight.train")
    
    logger.info("=" * 60)
    logger.info("Starting SwellSight Wave Analysis Training")
    logger.info("=" * 60)
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
        
        logger.info("[OK] Configuration loaded and validated")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = WaveAnalysisTrainer(config)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch, _ = trainer.load_checkpoint(args.resume)
        
        # Load data
        logger.info("Loading datasets...")
        try:
            from swellsight.data.datasets import WaveDataset
            from torch.utils.data import DataLoader
            
            # Get data configuration
            if hasattr(config, 'training'):
                batch_size = config.training.batch_size
                num_epochs = config.training.num_epochs
            else:
                batch_size = config.get('training', {}).get('batch_size', 32)
                num_epochs = config.get('training', {}).get('num_epochs', 100)
            
            if hasattr(config, 'data'):
                target_resolution = tuple(config.data.min_resolution)
            else:
                target_resolution = tuple(config.get('data', {}).get('target_resolution', [224, 224]))
            
            # Create datasets
            train_dataset = WaveDataset(
                data_dir=args.data_dir,
                split='train',
                train_ratio=0.8,
                target_resolution=target_resolution
            )
            
            val_dataset = WaveDataset(
                data_dir=args.data_dir,
                split='validation',
                train_ratio=0.8,
                target_resolution=target_resolution
            )
            
            # Check if we have data
            if len(train_dataset) == 0:
                logger.error(f"No training data found in {args.data_dir}")
                logger.error("Please ensure your data directory contains .npy files with corresponding _labels.npy files")
                logger.error("")
                logger.error("Expected structure:")
                logger.error("  data/")
                logger.error("    image_001.npy")
                logger.error("    image_001_labels.npy")
                logger.error("    image_002.npy")
                logger.error("    image_002_labels.npy")
                logger.error("    ...")
                return 1
            
            if len(val_dataset) == 0:
                logger.warning("No validation data found. Using 10% of training data for validation.")
                # Split training data
                train_size = int(0.9 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size]
                )
            
            logger.info(f"[OK] Training samples: {len(train_dataset)}")
            logger.info(f"[OK] Validation samples: {len(val_dataset)}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=torch.cuda.is_available()
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            logger.info(f"[OK] Data loaders created (batch_size={batch_size})")
            logger.info("")
            
            # Start training
            logger.info("=" * 60)
            logger.info("Starting Training")
            logger.info("=" * 60)
            trainer.train(train_loader, val_loader, num_epochs=num_epochs)
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("Training completed successfully!")
            logger.info("=" * 60)
            logger.info(f"Checkpoints saved to: {trainer.save_dir}")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.error("Make sure swellsight.data.datasets is available")
            return 1
        except Exception as e:
            logger.error(f"Error during data loading or training: {e}", exc_info=True)
            return 1
        
        logger.info("Training setup completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())