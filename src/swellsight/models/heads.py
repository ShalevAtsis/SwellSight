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
    """Regression head for wave height prediction (0.5-8.0m range) with dominant wave selection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__(input_dim, hidden_dim)
        
        # Main height regression network
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
        
        # Confidence estimation network
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Dominant wave detection network (helps identify primary wave vs secondary waves)
        self.dominance_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Score indicating how dominant this wave measurement is
        )
        
        # Wave count estimation (helps with multi-wave scenarios)
        self.wave_count_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU()  # Estimate number of significant waves (0+)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for wave height prediction with dominant wave analysis."""
        # Get raw predictions
        height_raw = self.regressor(x)
        confidence = self.confidence_head(x)
        dominance_score = self.dominance_head(x)
        wave_count_raw = self.wave_count_head(x)
        
        # Scale to wave height range [0.5, 8.0] meters
        height_meters = 0.5 + height_raw * 7.5
        
        # Process wave count (clamp to reasonable range)
        wave_count = torch.clamp(wave_count_raw, min=0.0, max=10.0)
        
        # Adjust confidence based on dominance and wave complexity
        # Higher dominance = more confident in single dominant wave
        # Lower wave count = more confident in measurement
        dominance_factor = dominance_score
        complexity_factor = torch.clamp(1.0 - (wave_count - 1.0) * 0.1, min=0.5, max=1.0)
        adjusted_confidence = confidence * dominance_factor * complexity_factor
        
        # Detect extreme conditions
        is_extreme = torch.logical_or(height_meters < 0.5, height_meters > 8.0)
        
        return {
            "height_meters": height_meters,
            "height_confidence": adjusted_confidence,
            "height_raw": height_raw,
            "dominance_score": dominance_score,
            "wave_count_estimate": wave_count,
            "is_extreme_height": is_extreme.float(),
            "base_confidence": confidence  # Original confidence before adjustments
        }

class DirectionHead(PredictionHead):
    """Classification head for wave direction (Left/Right/Straight) with mixed condition handling."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__(input_dim, hidden_dim)
        self.num_classes = num_classes
        self.class_names = ["LEFT", "RIGHT", "STRAIGHT"]
        
        # Main direction classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Mixed condition detector (detects when multiple directions are present)
        self.mixed_condition_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Score indicating mixed direction conditions
        )
        
        # Direction strength estimator (how strong/clear the direction is)
        self.direction_strength_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Strength of directional pattern
        )
        
        # Multiple wave train detector
        self.wave_train_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU()  # Number of distinct wave trains detected
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for direction classification with mixed condition analysis."""
        # Get main direction predictions
        logits = self.classifier(x)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class and base confidence
        base_confidence, predicted = torch.max(probabilities, dim=-1)
        
        # Get additional analysis
        mixed_score = self.mixed_condition_head(x)
        direction_strength = self.direction_strength_head(x)
        wave_train_count = torch.clamp(self.wave_train_head(x), min=1.0, max=5.0)
        
        # Adjust confidence based on mixed conditions and direction strength
        # Lower mixed score = more confident in single direction
        # Higher direction strength = more confident in classification
        mixed_penalty = mixed_score * 0.3  # Reduce confidence for mixed conditions
        strength_bonus = direction_strength * 0.2  # Increase confidence for strong patterns
        
        adjusted_confidence = base_confidence * (1.0 - mixed_penalty) * (1.0 + strength_bonus)
        adjusted_confidence = torch.clamp(adjusted_confidence, min=0.1, max=1.0)
        
        # Detect mixed conditions (high mixed score or low direction strength)
        is_mixed_conditions = torch.logical_or(mixed_score > 0.6, direction_strength < 0.3)
        
        # Calculate per-class confidence scores
        class_confidences = probabilities * adjusted_confidence.unsqueeze(-1)
        
        return {
            "direction_logits": logits,
            "direction_probabilities": probabilities,
            "direction_predicted": predicted,
            "direction_confidence": adjusted_confidence,
            "direction_class_confidences": class_confidences,
            "mixed_conditions_score": mixed_score,
            "direction_strength": direction_strength,
            "wave_train_count": wave_train_count,
            "is_mixed_conditions": is_mixed_conditions.float(),
            "base_confidence": base_confidence
        }

class BreakingTypeHead(PredictionHead):
    """Classification head for breaking type (Spilling/Plunging/Surging) with mixed pattern handling and No Breaking detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__(input_dim, hidden_dim)
        self.num_classes = num_classes
        self.class_names = ["SPILLING", "PLUNGING", "SURGING", "NO_BREAKING"]
        
        # Main breaking type classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Breaking intensity detector (how strong the breaking pattern is)
        self.breaking_intensity_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Intensity of breaking activity
        )
        
        # Mixed breaking pattern detector
        self.mixed_breaking_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Score indicating mixed breaking patterns
        )
        
        # Breaking clarity estimator (how clear/distinct the breaking pattern is)
        self.breaking_clarity_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Clarity of breaking pattern
        )
        
        # No breaking detector (specific detector for absence of breaking)
        self.no_breaking_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Probability of no breaking activity
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for breaking type classification with enhanced analysis."""
        # Get main breaking type predictions
        logits = self.classifier(x)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class and base confidence
        base_confidence, predicted = torch.max(probabilities, dim=-1)
        
        # Get additional breaking analysis
        breaking_intensity = self.breaking_intensity_head(x)
        mixed_breaking_score = self.mixed_breaking_head(x)
        breaking_clarity = self.breaking_clarity_head(x)
        no_breaking_score = self.no_breaking_head(x)
        
        # Adjust confidence based on breaking characteristics
        # Higher intensity and clarity = more confident
        # Lower mixed score = more confident in single type
        intensity_factor = breaking_intensity
        clarity_factor = breaking_clarity
        mixed_penalty = mixed_breaking_score * 0.2
        
        adjusted_confidence = base_confidence * intensity_factor * clarity_factor * (1.0 - mixed_penalty)
        adjusted_confidence = torch.clamp(adjusted_confidence, min=0.1, max=1.0)
        
        # Handle "No Breaking" detection
        # If no_breaking_score is high and breaking_intensity is low, favor NO_BREAKING
        no_breaking_threshold = 0.6
        low_intensity_threshold = 0.3
        
        should_be_no_breaking = torch.logical_and(
            no_breaking_score > no_breaking_threshold,
            breaking_intensity < low_intensity_threshold
        )
        
        # Override prediction if strong evidence for no breaking
        final_predicted = torch.where(
            should_be_no_breaking,
            torch.tensor(3, device=predicted.device),  # NO_BREAKING class index
            predicted
        )
        
        # Adjust confidence for no breaking cases
        final_confidence = torch.where(
            should_be_no_breaking,
            no_breaking_score * 0.9,  # High confidence for clear no-breaking cases
            adjusted_confidence
        )
        
        # Calculate percentage breakdown for mixed breaking types
        # Normalize probabilities excluding NO_BREAKING for breakdown
        breaking_probs = probabilities[:, :3]  # Only SPILLING, PLUNGING, SURGING
        breaking_probs_normalized = breaking_probs / (breaking_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Create percentage breakdown
        percentage_breakdown = breaking_probs_normalized * 100.0
        
        # Detect mixed breaking patterns
        is_mixed_breaking = torch.logical_or(
            mixed_breaking_score > 0.5,
            breaking_clarity < 0.4
        )
        
        return {
            "breaking_logits": logits,
            "breaking_probabilities": probabilities,
            "breaking_predicted": final_predicted,
            "breaking_confidence": final_confidence,
            "breaking_intensity": breaking_intensity,
            "mixed_breaking_score": mixed_breaking_score,
            "breaking_clarity": breaking_clarity,
            "no_breaking_score": no_breaking_score,
            "is_mixed_breaking": is_mixed_breaking.float(),
            "percentage_breakdown": percentage_breakdown,
            "base_confidence": base_confidence
        }