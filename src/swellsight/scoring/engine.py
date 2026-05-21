"""
Surf score engine v1 — weighted formula (0–100).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ScoreBreakdown:
    wave_quality: float
    size_factor: float
    confidence_factor: float
    safety_penalty: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class SurfScoreEngine:
    """
    Combines wave metrics into a single surf score.

    Inputs: height (m), direction, breaking, confidences, optional quality.
    """

    def __init__(
        self,
        ideal_height_m: float = 1.5,
        max_height_m: float = 3.0,
        min_confidence: float = 0.5,
    ):
        self.ideal_height_m = ideal_height_m
        self.max_height_m = max_height_m
        self.min_confidence = min_confidence

    def compute(
        self,
        height_meters: float,
        direction: str,
        breaking_type: str,
        height_confidence: float = 1.0,
        direction_confidence: float = 1.0,
        breaking_confidence: float = 1.0,
        overall_confidence: Optional[float] = None,
        extreme_conditions: bool = False,
        depth_quality: Optional[float] = None,
    ) -> tuple[int, ScoreBreakdown]:
        if overall_confidence is None:
            overall_confidence = (
                height_confidence + direction_confidence + breaking_confidence
            ) / 3.0

        # Size factor: peak near ideal_height, decay outside range
        if height_meters <= 0.3:
            size_factor = 0.2
        elif height_meters <= self.ideal_height_m:
            size_factor = 0.5 + 0.5 * (height_meters / self.ideal_height_m)
        elif height_meters <= self.max_height_m:
            size_factor = 1.0 - 0.3 * (
                (height_meters - self.ideal_height_m)
                / (self.max_height_m - self.ideal_height_m)
            )
        else:
            size_factor = max(0.3, 0.7 - 0.1 * (height_meters - self.max_height_m))

        # Breaking preference for surfability
        breaking_bonus = {
            "SPILLING": 1.0,
            "PLUNGING": 0.85,
            "SURGING": 0.7,
            "NO_BREAKING": 0.5,
        }.get(breaking_type.upper(), 0.75)

        wave_quality = size_factor * breaking_bonus
        if depth_quality is not None:
            wave_quality *= max(0.5, min(1.0, depth_quality))

        confidence_factor = max(self.min_confidence, min(1.0, overall_confidence))
        safety_penalty = 0.35 if extreme_conditions or height_meters > 4.0 else 0.0

        raw = (wave_quality * 0.55 + confidence_factor * 0.35) * 100.0
        raw *= 1.0 - safety_penalty
        surf_score = int(max(0, min(100, round(raw))))

        breakdown = ScoreBreakdown(
            wave_quality=round(wave_quality, 4),
            size_factor=round(size_factor, 4),
            confidence_factor=round(confidence_factor, 4),
            safety_penalty=round(safety_penalty, 4),
        )
        return surf_score, breakdown

    def compute_from_result(self, wave_result: Dict[str, Any]) -> tuple[int, Dict[str, float]]:
        wm = wave_result.get("wave_metrics", wave_result)
        score, breakdown = self.compute(
            height_meters=float(wm.get("wave_height_meters", wm.get("height_meters", 0))),
            direction=str(wm.get("direction", "STRAIGHT")),
            breaking_type=str(wm.get("breaking_type", "SPILLING")),
            height_confidence=float(wm.get("height_confidence", 0.8)),
            direction_confidence=float(wm.get("direction_confidence", 0.8)),
            breaking_confidence=float(wm.get("breaking_confidence", 0.8)),
            overall_confidence=float(wave_result.get("overall_confidence", 0.8)),
            extreme_conditions=bool(wm.get("extreme_conditions", False)),
        )
        return score, breakdown.to_dict()
