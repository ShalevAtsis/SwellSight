"""
Multi-task loss functions for balanced training.

Implements loss balancing across wave height regression, direction classification,
and breaking type classification tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class MultiTaskLoss(nn.Module):
    """Balanced multi-task loss for wave analysis."""
    
    def __init__(self, 
                 height_weight: float = 1.0,
                 direction_weight: float = 1.0, 
                 breaking_weight: float = 1.0,
                 adaptive_weighting: bool = True):
        """Initialize multi-task loss.
        
        Args:
            height_weight: Weight for height regression loss
            direction_weight: Weight for direction classification loss
            breaking_weight: Weight for breaking type classification loss
            adaptive_weighting: Whether to use adaptive loss weighting
        """
        super().__init__()
        
        self.height_weight = height_weight
        self.direction_weight = direction_weight
        self.breaking_weight = breaking_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Individual loss functions
        self.height_loss = nn.MSELoss()
        self.direction_loss = nn.CrossEntropyLoss()
        self.breaking_loss = nn.CrossEntropyLoss()
        
        # For adaptive weighting
        if adaptive_weighting:
            self.log_vars = nn.Parameter(torch.zeros(3))
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss.
        
        Args:
            predictions: Model predictions for all tasks
            targets: Ground truth targets for all tasks
            
        Returns:
            Dictionary with individual and total losses
        """
        # Individual task losses
        height_loss = self.height_loss(
            predictions["height_meters"], 
            targets["height_meters"]
        )
        
        direction_loss = self.direction_loss(
            predictions["direction_logits"],
            targets["direction_labels"]
        )
        
        breaking_loss = self.breaking_loss(
            predictions["breaking_logits"],
            targets["breaking_labels"]
        )
        
        if self.adaptive_weighting:
            # Adaptive uncertainty weighting (Kendall et al.)
            precision1 = torch.exp(-self.log_vars[0])
            precision2 = torch.exp(-self.log_vars[1])
            precision3 = torch.exp(-self.log_vars[2])
            
            total_loss = (
                precision1 * height_loss + self.log_vars[0] +
                precision2 * direction_loss + self.log_vars[1] +
                precision3 * breaking_loss + self.log_vars[2]
            )
        else:
            # Fixed weighting
            total_loss = (
                self.height_weight * height_loss +
                self.direction_weight * direction_loss +
                self.breaking_weight * breaking_loss
            )
        
        return {
            "total_loss": total_loss,
            "height_loss": height_loss,
            "direction_loss": direction_loss,
            "breaking_loss": breaking_loss,
            "height_weight": self.height_weight if not self.adaptive_weighting else precision1,
            "direction_weight": self.direction_weight if not self.adaptive_weighting else precision2,
            "breaking_weight": self.breaking_weight if not self.adaptive_weighting else precision3
        }