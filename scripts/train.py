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

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to dataset_manifest.json (optional)",
    )

    parser.add_argument(
        "--depth-dir",
        type=str,
        default=None,
        help="Directory with *_depth.npy files (default: data/depth_maps)",
    )

    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Build manifest from data-dir before training",
    )

    parser.add_argument(
        "--require-depth",
        action="store_true",
        help="Skip samples without depth maps",
    )

    parser.add_argument(
        "--sim-to-real",
        action="store_true",
        help="Run synthetic pretrain then real finetune (requires --synthetic-dir and --real-dir)",
    )

    parser.add_argument("--synthetic-dir", default="data/synthetic", help="Synthetic pretrain data")
    parser.add_argument("--real-dir", default="data", help="Real finetune data")
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")

    return parser.parse_args()


def _make_loaders(
    data_dir: str,
    batch_size: int,
    target_resolution,
    manifest_path,
    depth_dir: str,
    require_depth: bool,
):
    from swellsight.data.datasets import WaveDataset
    from torch.utils.data import DataLoader

    train_dataset = WaveDataset(
        data_dir=data_dir,
        split="train",
        train_ratio=0.8,
        target_resolution=target_resolution,
        manifest_path=manifest_path,
        depth_dir=depth_dir,
        require_depth=require_depth,
    )
    val_dataset = WaveDataset(
        data_dir=data_dir,
        split="validation",
        train_ratio=0.8,
        target_resolution=target_resolution,
        manifest_path=manifest_path,
        depth_dir=depth_dir,
        require_depth=require_depth,
    )
    if len(val_dataset) == 0 and len(train_dataset) > 0:
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

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
        log_dir = str(output_dir / "logs")
        trainer = WaveAnalysisTrainer(
            config,
            log_dir=None if args.no_tensorboard else log_dir,
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            start_epoch, _ = trainer.load_checkpoint(args.resume)
        
        # Load data
        logger.info("Loading datasets...")
        try:
            # Get data configuration
            if hasattr(config, 'training'):
                batch_size = config.training.batch_size
                num_epochs = config.training.num_epochs
            else:
                batch_size = config.get('training', {}).get('batch_size', 32)
                num_epochs = config.get('training', {}).get('num_epochs', 100)
            
            if hasattr(config, 'data'):
                target_resolution = tuple(config.data.target_resolution)
            else:
                target_resolution = tuple(config.get('data', {}).get('target_resolution', [518, 518]))

            manifest_path = args.manifest
            if args.build_manifest:
                from swellsight.data.manifest import build_manifest
                manifest_path = str(Path(args.data_dir).parent / "manifests" / "dataset_manifest.json")
                build_manifest(args.data_dir, manifest_path)
                logger.info("Built manifest: %s", manifest_path)

            depth_dir = args.depth_dir or str(Path("data/depth_maps"))

            if args.sim_to_real:
                from swellsight.mlops.experiment import ExperimentLogger

                exp = ExperimentLogger()
                exp.start("sim-to-real", {"config": args.config})
                exp.log_params(
                    {
                        "synthetic_dir": args.synthetic_dir,
                        "real_dir": args.real_dir,
                    }
                )

                syn_train, syn_val, n_syn, _ = _make_loaders(
                    args.synthetic_dir,
                    batch_size,
                    target_resolution,
                    manifest_path,
                    depth_dir,
                    args.require_depth,
                )
                real_train, real_val, n_real, _ = _make_loaders(
                    args.real_dir,
                    batch_size,
                    target_resolution,
                    manifest_path,
                    depth_dir,
                    args.require_depth,
                )
                if n_syn == 0 or n_real == 0:
                    logger.error("Sim-to-real needs data in both synthetic (%s) and real (%s) dirs", args.synthetic_dir, args.real_dir)
                    return 1

                result = trainer.train_sim_to_real(
                    syn_train,
                    syn_val,
                    real_train,
                    real_val,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                )
                exp.log_metrics(trainer.best_metrics)
                logger.info("Sim-to-real result: %s", result)
            else:
                train_loader, val_loader, n_train, n_val = _make_loaders(
                    args.data_dir,
                    batch_size,
                    target_resolution,
                    manifest_path,
                    depth_dir,
                    args.require_depth,
                )
                if n_train == 0:
                    logger.error("No training data in %s", args.data_dir)
                    return 1
                logger.info("[OK] Training samples: %s | Validation: %s", n_train, n_val)
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