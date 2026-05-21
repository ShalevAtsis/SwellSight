from swellsight.scoring.engine import SurfScoreEngine


def test_surf_score_in_valid_range():
    engine = SurfScoreEngine()
    score, breakdown = engine.compute(
        height_meters=1.5,
        direction="RIGHT",
        breaking_type="SPILLING",
        height_confidence=0.9,
        direction_confidence=0.9,
        breaking_confidence=0.85,
    )
    assert 0 <= score <= 100
    assert breakdown.size_factor > 0


def test_extreme_conditions_lower_score():
    engine = SurfScoreEngine()
    normal, _ = engine.compute(1.5, "RIGHT", "SPILLING", 0.9, 0.9, 0.9, extreme_conditions=False)
    extreme, _ = engine.compute(5.0, "STRAIGHT", "PLUNGING", 0.9, 0.9, 0.9, extreme_conditions=True)
    assert extreme < normal
