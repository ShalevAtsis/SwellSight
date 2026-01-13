"""
Learning rate scheduling for wave analysis training.

Implements various scheduling strategies for optimal convergence.
"""

from typing import Any, Optional
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR, ReduceLROnPlateau
import math

class WarmupCosineAnnealingLR(_LRScheduler):
    """Cosine annealing scheduler with linear warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 min_lr: float = 0, last_epoch: int = -1):
        """Initialize warmup cosine annealing scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

def create_lr_scheduler(optimizer: optim.Optimizer, 
                       scheduler_type: str = "cosine",
                       num_epochs: int = 100,
                       warmup_epochs: int = 5,
                       min_lr: float = 1e-6,
                       step_size: int = 30,
                       step_gamma: float = 0.1,
                       plateau_patience: int = 10,
                       plateau_factor: float = 0.5,
                       **kwargs) -> Optional[_LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("cosine", "step", "plateau", "warmup_cosine")
        num_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs (for warmup_cosine)
        min_lr: Minimum learning rate (for cosine schedulers)
        step_size: Step size for StepLR
        step_gamma: Gamma factor for StepLR
        plateau_patience: Patience for ReduceLROnPlateau
        plateau_factor: Factor for ReduceLROnPlateau
        **kwargs: Additional scheduler parameters
        
    Returns:
        Configured learning rate scheduler or None
    """
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs, 
            eta_min=min_lr
        )
    
    elif scheduler_type == "warmup_cosine":
        return WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=num_epochs,
            min_lr=min_lr
        )
    
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=step_gamma
        )
    
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            verbose=True
        )
    
    elif scheduler_type == "none" or scheduler_type is None:
        return None
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")