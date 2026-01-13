"""
Multi-task training logic for wave analysis models.

Implements sim-to-real training strategy with balanced loss weighting.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import logging

from ..utils.config import TrainingConfig

class WaveAnalysisTrainer:
    """Trainer for multi-task wave analysis models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 device: torch.device = None,
                 synthetic_loader: Optional[DataLoader] = None,
                 real_loader: Optional[DataLoader] = None):
        """Initialize wave analysis trainer.
        
        Args:
            model: Multi-task wave analysis model
            train_loader: Training data loader (combined or synthetic-only)
            val_loader: Validation data loader
            config: Training configuration
            device: Training device (GPU/CPU)
            synthetic_loader: Synthetic data loader for sim-to-real training
            real_loader: Real data loader for sim-to-real training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sim-to-real training loaders
        self.synthetic_loader = synthetic_loader
        self.real_loader = real_loader
        
        self.logger = logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        self.training_phase = "pretrain"  # "pretrain", "finetune"
        
        self._setup_training()
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and mixed precision."""
        from ..models.losses import MultiTaskLoss
        from .scheduler import create_lr_scheduler
        
        # Initialize multi-task loss with adaptive weighting
        self.criterion = MultiTaskLoss(
            height_weight=self.config.height_loss_weight,
            direction_weight=self.config.direction_loss_weight,
            breaking_weight=self.config.breaking_loss_weight,
            adaptive_weighting=self.config.adaptive_loss_weighting
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler_type,
            num_epochs=self.config.num_epochs,
            warmup_epochs=self.config.warmup_epochs
        )
        
        # Setup mixed precision training
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize training metrics tracking
        self.loss_history = {
            'train_total': [],
            'train_height': [],
            'train_direction': [],
            'train_breaking': [],
            'val_total': [],
            'val_height': [],
            'val_direction': [],
            'val_breaking': [],
            'loss_weights': [],
            'training_phase': []
        }
        
        self.logger.info("Training setup completed successfully")
    
    def train_sim_to_real(self) -> Dict[str, Any]:
        """Execute complete sim-to-real training strategy.
        
        Returns:
            Training history and final metrics
        """
        self.logger.info("Starting sim-to-real training strategy")
        
        # Phase 1: Pre-training on synthetic data
        if self.synthetic_loader:
            self.logger.info(f"Phase 1: Pre-training on synthetic data for {self.config.pretrain_epochs} epochs")
            self.training_phase = "pretrain"
            self.train_loader = self.synthetic_loader
            
            pretrain_results = self._train_phase(
                num_epochs=self.config.pretrain_epochs,
                phase_name="pretrain"
            )
            
            # Save pre-trained model
            self.save_checkpoint("pretrained_model.pth", is_best=False)
            self.logger.info("Pre-training phase completed")
        
        # Phase 2: Fine-tuning on real data
        if self.real_loader:
            self.logger.info(f"Phase 2: Fine-tuning on real data for {self.config.finetune_epochs} epochs")
            self.training_phase = "finetune"
            
            # Reduce learning rate for fine-tuning
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * 0.1
            
            # Use real data or mixed data for fine-tuning
            if hasattr(self.config, 'use_mixed_finetuning') and self.config.use_mixed_finetuning:
                # Create mixed dataset with specified ratios
                self.train_loader = self._create_mixed_loader()
            else:
                self.train_loader = self.real_loader
            
            # Reset epoch counter for fine-tuning
            self.current_epoch = 0
            
            finetune_results = self._train_phase(
                num_epochs=self.config.finetune_epochs,
                phase_name="finetune"
            )
            
            self.logger.info("Fine-tuning phase completed")
        
        # Combine results
        final_results = {
            'sim_to_real_completed': True,
            'pretrain_epochs': self.config.pretrain_epochs,
            'finetune_epochs': self.config.finetune_epochs,
            'best_val_loss': self.best_val_loss,
            'training_history': self.loss_history
        }
        
        return final_results
    
    def _create_mixed_loader(self) -> DataLoader:
        """Create mixed data loader with synthetic and real data.
        
        Returns:
            DataLoader with mixed synthetic and real data
        """
        if not self.synthetic_loader or not self.real_loader:
            raise ValueError("Both synthetic and real loaders required for mixed training")
        
        # Calculate dataset sizes based on ratios
        synthetic_size = int(len(self.synthetic_loader.dataset) * self.config.synthetic_data_ratio)
        real_size = int(len(self.real_loader.dataset) * self.config.real_data_ratio)
        
        # Create subset datasets
        synthetic_subset = torch.utils.data.Subset(
            self.synthetic_loader.dataset, 
            torch.randperm(len(self.synthetic_loader.dataset))[:synthetic_size]
        )
        real_subset = torch.utils.data.Subset(
            self.real_loader.dataset,
            torch.randperm(len(self.real_loader.dataset))[:real_size]
        )
        
        # Combine datasets
        mixed_dataset = ConcatDataset([synthetic_subset, real_subset])
        
        # Create mixed loader
        mixed_loader = DataLoader(
            mixed_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory
        )
        
        self.logger.info(f"Created mixed loader: {synthetic_size} synthetic + {real_size} real samples")
        return mixed_loader
    
    def _train_phase(self, num_epochs: int, phase_name: str) -> Dict[str, Any]:
        """Train for a specific phase (pretrain or finetune).
        
        Args:
            num_epochs: Number of epochs for this phase
            phase_name: Name of the training phase
            
        Returns:
            Phase training results
        """
        phase_best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training step
            train_metrics = self.train_epoch()
            train_metrics['training_phase'] = phase_name
            
            # Validation step
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate_epoch()
                val_metrics['training_phase'] = phase_name
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Track loss weights for adaptive weighting
                if hasattr(self.criterion, 'log_vars'):
                    weights = {
                        'height_weight': torch.exp(-self.criterion.log_vars[0]).item(),
                        'direction_weight': torch.exp(-self.criterion.log_vars[1]).item(),
                        'breaking_weight': torch.exp(-self.criterion.log_vars[2]).item()
                    }
                    epoch_metrics.update(weights)
                    self.loss_history['loss_weights'].append(weights)
                
                # Update training history
                for key, value in epoch_metrics.items():
                    if key in self.loss_history:
                        self.loss_history[key].append(value)
                
                # Track training phase
                self.loss_history['training_phase'].append(phase_name)
                
                # Check for improvement
                current_val_loss = val_metrics['val_total_loss']
                if current_val_loss < phase_best_loss:
                    phase_best_loss = current_val_loss
                    patience_counter = 0
                    
                    # Update global best if this is better
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.save_checkpoint(
                            f"best_model_{phase_name}_epoch_{epoch}.pth", 
                            is_best=True
                        )
                        self.logger.info(f"New best model saved at {phase_name} epoch {epoch}")
                else:
                    patience_counter += 1
                
                # Log epoch results
                self.logger.info(
                    f"{phase_name.capitalize()} Epoch {epoch}: "
                    f"Train Loss={train_metrics['train_total_loss']:.4f}, "
                    f"Val Loss={current_val_loss:.4f}, "
                    f"Val Height MAE={val_metrics['val_height_mae']:.3f}m, "
                    f"Val Direction Acc={val_metrics['val_direction_accuracy']:.3f}, "
                    f"Val Breaking Acc={val_metrics['val_breaking_accuracy']:.3f}"
                )
                
                # Early stopping check
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at {phase_name} epoch {epoch}")
                    break
            else:
                # Only training metrics for this epoch
                train_metrics['training_phase'] = phase_name
                for key, value in train_metrics.items():
                    if key in self.loss_history:
                        self.loss_history[key].append(value)
                self.loss_history['training_phase'].append(phase_name)
            
            # Save checkpoint periodically
            if epoch % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint_{phase_name}_epoch_{epoch}.pth")
        
        return {
            'phase': phase_name,
            'epochs_completed': epoch + 1,
            'best_loss': phase_best_loss
        }
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and mixed precision."""
        from ..models.losses import MultiTaskLoss
        from .scheduler import create_lr_scheduler
        
        # Initialize multi-task loss with adaptive weighting
        self.criterion = MultiTaskLoss(
            height_weight=self.config.height_loss_weight,
            direction_weight=self.config.direction_loss_weight,
            breaking_weight=self.config.breaking_loss_weight,
            adaptive_weighting=self.config.adaptive_loss_weighting
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler_type,
            num_epochs=self.config.num_epochs,
            warmup_epochs=self.config.warmup_epochs
        )
        
        # Setup mixed precision training
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize training metrics tracking
        self.loss_history = {
            'train_total': [],
            'train_height': [],
            'train_direction': [],
            'train_breaking': [],
            'val_total': [],
            'val_height': [],
            'val_direction': [],
            'val_breaking': [],
            'loss_weights': []
        }
        
        self.logger.info("Training setup completed successfully")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'height': 0.0,
            'direction': 0.0,
            'breaking': 0.0,
            'count': 0
        }
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            rgb_images = batch['rgb_image'].to(self.device)
            depth_maps = batch['depth_map'].to(self.device)
            targets = {
                'height_meters': batch['height_meters'].to(self.device),
                'direction_labels': batch['direction_labels'].to(self.device),
                'breaking_labels': batch['breaking_labels'].to(self.device)
            }
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(rgb_images, depth_maps)
                    loss_dict = self.criterion(predictions, targets)
                    total_loss = loss_dict['total_loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                predictions = self.model(rgb_images, depth_maps)
                loss_dict = self.criterion(predictions, targets)
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update learning rate scheduler (if step-based)
            if hasattr(self.scheduler, 'step') and self.config.scheduler_step_on_batch:
                self.scheduler.step()
            
            # Accumulate losses
            epoch_losses['total'] += loss_dict['total_loss'].item()
            epoch_losses['height'] += loss_dict['height_loss'].item()
            epoch_losses['direction'] += loss_dict['direction_loss'].item()
            epoch_losses['breaking'] += loss_dict['breaking_loss'].item()
            epoch_losses['count'] += 1
            
            # Log batch progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                    f"Loss={total_loss.item():.4f}, "
                    f"Height={loss_dict['height_loss'].item():.4f}, "
                    f"Direction={loss_dict['direction_loss'].item():.4f}, "
                    f"Breaking={loss_dict['breaking_loss'].item():.4f}"
                )
        
        # Calculate average losses
        avg_losses = {
            'train_total_loss': epoch_losses['total'] / epoch_losses['count'],
            'train_height_loss': epoch_losses['height'] / epoch_losses['count'],
            'train_direction_loss': epoch_losses['direction'] / epoch_losses['count'],
            'train_breaking_loss': epoch_losses['breaking'] / epoch_losses['count']
        }
        
        # Update learning rate scheduler (if epoch-based)
        if hasattr(self.scheduler, 'step') and not self.config.scheduler_step_on_batch:
            self.scheduler.step()
        
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        epoch_losses = {
            'total': 0.0,
            'height': 0.0,
            'direction': 0.0,
            'breaking': 0.0,
            'count': 0
        }
        
        # Additional metrics for validation
        height_mae = 0.0
        direction_correct = 0
        breaking_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                rgb_images = batch['rgb_image'].to(self.device)
                depth_maps = batch['depth_map'].to(self.device)
                targets = {
                    'height_meters': batch['height_meters'].to(self.device),
                    'direction_labels': batch['direction_labels'].to(self.device),
                    'breaking_labels': batch['breaking_labels'].to(self.device)
                }
                
                # Forward pass
                if self.config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(rgb_images, depth_maps)
                        loss_dict = self.criterion(predictions, targets)
                else:
                    predictions = self.model(rgb_images, depth_maps)
                    loss_dict = self.criterion(predictions, targets)
                
                # Accumulate losses
                epoch_losses['total'] += loss_dict['total_loss'].item()
                epoch_losses['height'] += loss_dict['height_loss'].item()
                epoch_losses['direction'] += loss_dict['direction_loss'].item()
                epoch_losses['breaking'] += loss_dict['breaking_loss'].item()
                epoch_losses['count'] += 1
                
                # Calculate additional metrics
                batch_size = rgb_images.size(0)
                total_samples += batch_size
                
                # Height MAE
                height_mae += torch.abs(
                    predictions['height_meters'] - targets['height_meters']
                ).sum().item()
                
                # Direction accuracy
                direction_pred = torch.argmax(predictions['direction_logits'], dim=1)
                direction_correct += (direction_pred == targets['direction_labels']).sum().item()
                
                # Breaking type accuracy
                breaking_pred = torch.argmax(predictions['breaking_logits'], dim=1)
                breaking_correct += (breaking_pred == targets['breaking_labels']).sum().item()
        
        # Calculate average losses and metrics
        avg_losses = {
            'val_total_loss': epoch_losses['total'] / epoch_losses['count'],
            'val_height_loss': epoch_losses['height'] / epoch_losses['count'],
            'val_direction_loss': epoch_losses['direction'] / epoch_losses['count'],
            'val_breaking_loss': epoch_losses['breaking'] / epoch_losses['count'],
            'val_height_mae': height_mae / total_samples,
            'val_direction_accuracy': direction_correct / total_samples,
            'val_breaking_accuracy': breaking_correct / total_samples
        }
        
        return avg_losses
    
    def train(self) -> Dict[str, Any]:
        """Complete training loop.
        
        Returns:
            Training history and final metrics
        """
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            if epoch % self.config.validate_every == 0:
                val_metrics = self.validate_epoch()
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                
                # Track loss weights for adaptive weighting
                if hasattr(self.criterion, 'log_vars'):
                    weights = {
                        'height_weight': torch.exp(-self.criterion.log_vars[0]).item(),
                        'direction_weight': torch.exp(-self.criterion.log_vars[1]).item(),
                        'breaking_weight': torch.exp(-self.criterion.log_vars[2]).item()
                    }
                    epoch_metrics.update(weights)
                    self.loss_history['loss_weights'].append(weights)
                
                # Update training history
                for key, value in epoch_metrics.items():
                    if key in self.loss_history:
                        self.loss_history[key].append(value)
                
                # Check for improvement
                current_val_loss = val_metrics['val_total_loss']
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        f"best_model_epoch_{epoch}.pth", 
                        is_best=True
                    )
                    self.logger.info(f"New best model saved at epoch {epoch}")
                else:
                    patience_counter += 1
                
                # Log epoch results
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['train_total_loss']:.4f}, "
                    f"Val Loss={current_val_loss:.4f}, "
                    f"Val Height MAE={val_metrics['val_height_mae']:.3f}m, "
                    f"Val Direction Acc={val_metrics['val_direction_accuracy']:.3f}, "
                    f"Val Breaking Acc={val_metrics['val_breaking_accuracy']:.3f}"
                )
                
                # Early stopping check
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                # Only training metrics for this epoch
                for key, value in train_metrics.items():
                    if key in self.loss_history:
                        self.loss_history[key].append(value)
            
            # Save checkpoint periodically
            if epoch % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
        
        # Final training summary
        final_metrics = {
            'total_epochs': self.current_epoch + 1,
            'best_val_loss': best_val_loss,
            'training_history': self.loss_history
        }
        
        self.logger.info("Training completed successfully")
        return final_metrics
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        import os
        from pathlib import Path
        
        # Ensure checkpoint directory exists
        checkpoint_path = Path(filepath)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.loss_history,
            'config': self.config,
            'is_best': is_best
        }
        
        # Add loss weights if using adaptive weighting
        if hasattr(self.criterion, 'log_vars'):
            checkpoint['loss_weights'] = {
                'height_weight': torch.exp(-self.criterion.log_vars[0]).item(),
                'direction_weight': torch.exp(-self.criterion.log_vars[1]).item(),
                'breaking_weight': torch.exp(-self.criterion.log_vars[2]).item()
            }
        
        try:
            torch.save(checkpoint, filepath)
            self.logger.info(f"Checkpoint saved: {filepath}")
            
            # Create symlink for best model
            if is_best:
                best_model_path = checkpoint_path.parent / "best_model.pth"
                if best_model_path.exists():
                    best_model_path.unlink()
                try:
                    best_model_path.symlink_to(checkpoint_path.name)
                except OSError:
                    # Fallback: copy file if symlink fails (Windows compatibility)
                    import shutil
                    shutil.copy2(filepath, best_model_path)
                    
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {filepath}: {e}")
            raise
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Checkpoint metadata
        """
        from pathlib import Path
        
        checkpoint_path = Path(filepath)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Restore model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler state
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore scaler state for mixed precision
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Restore training state
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            # Restore training history
            if 'training_history' in checkpoint:
                self.loss_history = checkpoint['training_history']
            
            # Restore loss weights if using adaptive weighting
            if 'loss_weights' in checkpoint and hasattr(self.criterion, 'log_vars'):
                weights = checkpoint['loss_weights']
                self.criterion.log_vars[0].data.fill_(-torch.log(torch.tensor(weights['height_weight'])))
                self.criterion.log_vars[1].data.fill_(-torch.log(torch.tensor(weights['direction_weight'])))
                self.criterion.log_vars[2].data.fill_(-torch.log(torch.tensor(weights['breaking_weight'])))
            
            self.logger.info(f"Checkpoint loaded: {filepath} (epoch {self.current_epoch})")
            
            # Return metadata
            return {
                'epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'is_best': checkpoint.get('is_best', False),
                'config': checkpoint.get('config', None)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {filepath}: {e}")
            raise