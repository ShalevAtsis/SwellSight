# Model card: wave-v0.1.0

| Field | Value |
|-------|--------|
| **Version** | wave-v0.1.0 |
| **Backbone** | DINOv2 ViT-B/14 |
| **Input** | RGB + depth (4 channels), 518×518 |
| **Tasks** | Height (m), direction (3-class), breaking (3-class) |
| **Checkpoint** | `models/promoted/wave-v0.1.0.pth` |

## Intended use

Beach cam images for recreational surf condition estimation. Not for navigation or safety-critical decisions.

## Limitations

- Trained primarily on synthetic + limited real data
- Performance degrades in heavy fog, night, or non-beach scenes
- Surf score is a heuristic v1 formula, not learned from user ratings

## Metrics (fill after training)

| Metric | Value |
|--------|-------|
| Height MAE | _TBD_ |
| Direction accuracy | _TBD_ |
| Breaking accuracy | _TBD_ |

## Promotion

```bash
python scripts/promote_model.py --version wave-v0.1.0 --checkpoint checkpoints/best_model.pth
```
