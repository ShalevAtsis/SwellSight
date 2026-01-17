import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
from typing import Dict, Any, Optional, Union

# Import the model we defined in the previous step
from swellsight.models.wave_model import WaveAnalysisModel

class WaveAnalysisTrainer:
    """
    Trainer for the SwellSight Wave Analysis Model.
    Handles the training loop, loss calculation, and checkpointing.
    """
    
    def __init__(self, config: Union[Dict[str, Any], Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dict or SwellSightConfig object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Handle both dict and config object
        if hasattr(config, 'training'):
            # Config object (SwellSightConfig)
            train_conf = config.training
            log_conf = config.system
            self.batch_size = train_conf.batch_size
            self.learning_rate = train_conf.learning_rate
            self.num_epochs = train_conf.num_epochs
            self.weight_decay = train_conf.weight_decay
            self.optimizer_name = 'AdamW'
            self.save_dir = Path(log_conf.output_dir) / 'checkpoints'
            self.save_frequency = train_conf.save_checkpoint_every
            
            # Loss weights
            self.weights = {
                'height': train_conf.height_loss_weight,
                'direction': train_conf.direction_loss_weight,
                'breaking_type': train_conf.breaking_loss_weight
            }
        else:
            # Dict config
            train_conf = config.get('training', {})
            log_conf = config.get('logging', {})
            self.batch_size = train_conf.get('batch_size', 32)
            self.learning_rate = float(train_conf.get('learning_rate', 1e-4))
            self.num_epochs = train_conf.get('num_epochs', 100)
            self.weight_decay = float(train_conf.get('weight_decay', 0.01))
            self.optimizer_name = train_conf.get('optimizer', 'AdamW')
            self.save_dir = Path(log_conf.get('save_dir', 'models/checkpoints'))
            self.save_frequency = train_conf.get('save_checkpoint_every', 5)
            
            # Loss weights
            self.weights = train_conf.get('loss_weights', 
                                        {'height': 1.0, 'direction': 1.0, 'breaking_type': 1.0})
        
        # 1. Setup Device & Directories
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Initialize Model
        self.logger.info(f"Initializing model on {self.device}...")
        self.model = WaveAnalysisModel(config).to(self.device)
        
        # 3. Define Loss Functions
        # Height = Regression (MSE)
        self.height_loss = nn.MSELoss()
        # Direction & Breaking = Classification (CrossEntropy)
        self.direction_loss = nn.CrossEntropyLoss()
        self.breaking_loss = nn.CrossEntropyLoss()
        
        # 4. Setup Optimizer
        # Filter parameters to only optimize those that require gradients 
        # (This handles the 'freeze_backbone' setting automatically)
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(params, lr=self.learning_rate)
            
        self.logger.info(f"[OK] Model initialized. Trainable parameters: {sum(p.numel() for p in params)}")
        self.logger.info(f"[OK] Device: {self.device}")
        self.logger.info(f"[OK] Optimizer: {self.optimizer_name}, LR: {self.learning_rate}")

    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None):
        """
        Main training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs (overrides config if provided)
        """
        if num_epochs is None:
            num_epochs = self.num_epochs
            
        best_val_loss = float('inf')
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # --- Training Phase ---
            train_metrics = self._run_epoch(train_loader, is_training=True)
            self._log_metrics(train_metrics, "Train")
            
            # --- Validation Phase ---
            val_metrics = self._run_epoch(val_loader, is_training=False)
            self._log_metrics(val_metrics, "Val")
            
            # --- Checkpointing ---
            current_val_loss = val_metrics['total_loss']
            
            # Save if Best
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                self.logger.info(f"[BEST] New Best Model Saved (Loss: {best_val_loss:.4f})")
            
            # Save Periodic
            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
        
        self.logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")

    def _run_epoch(self, loader, is_training):
        """Runs a single epoch of training or validation."""
        if is_training:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0
        height_losses = []
        dir_accs = []
        break_accs = []
        
        # Use TQDM for progress bar
        pbar = tqdm(loader, desc="Training" if is_training else "Validating", leave=False)
        
        for batch in pbar:
            # 1. Unpack Batch & Move to Device
            inputs = batch['input'].to(self.device) # Shape: (B, 4, H, W)
            labels = batch['labels']
            
            h_target = labels['height'].to(self.device).view(-1, 1) # Regression target (B, 1)
            d_target = labels['direction'].to(self.device)          # Class index (B,)
            b_target = labels['breaking_type'].to(self.device)      # Class index (B,)
            
            # 2. Zero Gradients
            if is_training:
                self.optimizer.zero_grad()
            
            # 3. Forward Pass
            with torch.set_grad_enabled(is_training):
                outputs = self.model(inputs)
                
                # 4. Calculate Losses
                loss_h = self.height_loss(outputs['height'], h_target)
                loss_d = self.direction_loss(outputs['direction'], d_target)
                loss_b = self.breaking_loss(outputs['breaking_type'], b_target)
                
                # Weighted Sum
                loss = (self.weights['height'] * loss_h + 
                        self.weights['direction'] * loss_d + 
                        self.weights['breaking_type'] * loss_b)
                
                # 5. Backward Pass
                if is_training:
                    loss.backward()
                    self.optimizer.step()
            
            # 6. Update Metrics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            height_losses.append(loss_h.item())
            
            # Calculate Accuracies
            _, d_pred = torch.max(outputs['direction'], 1)
            dir_acc = (d_pred == d_target).float().mean().item()
            dir_accs.append(dir_acc)
            
            _, b_pred = torch.max(outputs['breaking_type'], 1)
            break_acc = (b_pred == b_target).float().mean().item()
            break_accs.append(break_acc)
            
            # Update Progress Bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Aggregate metrics over epoch
        avg_loss = total_loss / len(loader.dataset)
        return {
            'total_loss': avg_loss,
            'height_mse': np.mean(height_losses),
            'direction_acc': np.mean(dir_accs),
            'breaking_acc': np.mean(break_accs)
        }

    def _log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Print metrics to console and logger."""
        msg = (f"  {prefix} Loss: {metrics['total_loss']:.4f} | "
               f"Height MSE: {metrics['height_mse']:.4f} | "
               f"Dir Acc: {metrics['direction_acc']:.2%} | "
               f"Brk Acc: {metrics['breaking_acc']:.2%}")
        self.logger.info(msg)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint to disk."""
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth"
        path = self.save_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint from disk.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        # Use weights_only=False for backward compatibility with older checkpoints
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        self.logger.info(f"[OK] Checkpoint loaded (Epoch {epoch+1})")
        self.logger.info(f"  Metrics: {metrics}")
        
        return epoch, metrics