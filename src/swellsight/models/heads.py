"""
Multi-task prediction heads for wave analysis.

Specialized heads for wave height regression, direction classification,
and breaking type classification.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class PredictionHead(nn.Module):
    """Base class for prediction heads."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

class WaveHeightHead(PredictionHead):
    """Regression head for wave height prediction (0.5-8.0m range)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__(input_dim, hidden_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output [0,1], will be scaled to [0.5, 8.0]
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for wave height prediction."""
        height_raw = self.regressor(x)
        confidence = self.confidence_head(x)
        
        # Scale to wave height range [0.5, 8.0] meters
        height_meters = 0.5 + height_raw * 7.5
        
        return {
            "height_meters": height_meters,
            "height_confidence": confidence
        }

class DirectionHead(PredictionHead):
    """Classification head for wave direction (Left/Right/Straight)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__(input_dim, hidden_dim)
        self.num_classes = num_classes
        self.class_names = ["LEFT", "RIGHT", "STRAIGHT"]
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for direction classification."""
        logits = self.classifier(x)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class and confidence
        confidence, predicted = torch.max(probabilities, dim=-1)
        
        return {
            "direction_logits": logits,
            "direction_probabilities": probabilities,
            "direction_predicted": predicted,
            "direction_confidence": confidence
        }

class BreakingTypeHead(PredictionHead):
    """Classification head for breaking type (Spilling/Plunging/Surging)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__(input_dim, hidden_dim)
        self.num_classes = num_classes
        self.class_names = ["SPILLING", "PLUNGING", "SURGING"]
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for breaking type classification."""
        logits = self.classifier(x)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class and confidence
        confidence, predicted = torch.max(probabilities, dim=-1)
        
        return {
            "breaking_logits": logits,
            "breaking_probabilities": probabilities,
            "breaking_predicted": predicted,
            "breaking_confidence": confidence
        }