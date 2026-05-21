# SwellSight Model Guide

Single reference for **hardware requirements**, **data layout**, **training**, and **inference** for the wave analysis model (Stage C: DINOv2 multi-task analyzer).

Phase 2 (user accounts, surf score UI, production API) builds on this; get the model path solid first.

---

## Architecture (one model, two entry points)

| Component | Module | Role |
|-----------|--------|------|
| Backbone | `src/swellsight/models/backbone.py` | DINOv2 + 4-channel (RGB+depth) adapter |
| Heads | `src/swellsight/models/heads.py` | Height, direction, breaking |
| Training model | `src/swellsight/models/wave_model.py` | `WaveAnalysisModel` |
| Inference analyzer | `src/swellsight/core/wave_analyzer.py` | `DINOv2WaveAnalyzer` (same weights) |
| Full pipeline | `src/swellsight/core/pipeline.py` | Depth-Anything-V2 → wave analyzer |
| Trainer | `src/swellsight/training/trainer.py` | Loop, checkpoints, metrics |

**Important:** Training saves `checkpoints/best_model.pth`. Inference loads that file into `DINOv2WaveAnalyzer` via `--checkpoint` (shared `state_dict` keys).

Depth extraction and FLUX synthetic generation are **separate** stages; see [TRAINING_FROM_SCRATCH.md](TRAINING_FROM_SCRATCH.md) for the full sim-to-real pipeline.

---

## Requirements

### Hardware

| Use case | GPU VRAM | RAM | Disk |
|----------|----------|-----|------|
| Training (recommended) | 8 GB+ (RTX 2070 or better) | 16 GB+ | 100 GB+ free |
| Inference only | 6 GB+ | 8 GB+ | 10 GB+ (models cache on first run) |
| CPU-only | — | 32 GB+ | Same (very slow) |

### Software

- **Python 3.9–3.11** (3.12+ may work; 3.13 is untested)
- CUDA-enabled PyTorch when using GPU
- Install:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e ".[training]"
# Inference + API:
pip install -e ".[inference]"
```

### Check readiness

```bash
python scripts/check_training_readiness.py
# or
swellsight check
```

---

## Data layout for training

Prepared samples go under a single directory (default `data/`). Each sample is a 4-channel-ready RGB array plus labels:

```
data/
  sample_001.npy          # RGB image array (H, W, 3), uint8 or float
  sample_001_labels.npy   # dict: height, direction, breaking_type
  sample_002.npy
  sample_002_labels.npy
  ...
```

**Label dict** (`*_labels.npy`, loaded with `allow_pickle=True`):

```python
{
    "height": 1.8,              # meters (float)
    "direction": "RIGHT",       # LEFT | RIGHT | STRAIGHT
    "breaking_type": "PLUNGING" # SPILLING | PLUNGING | SURGING
}
```

Optional pipeline folders (for full training workflow):

```
data/raw/beach_cams/     # Original images
data/depth_maps/         # From extract_depth_maps.py
data/synthetic/          # From generate_synthetic_data.py
data/augmented/          # From augment_data.py
```

Generate **dummy data** for a smoke test:

```bash
python scripts/generate_dummy_data.py --output-dir data
```

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Base model, training, paths |
| `configs/training.yaml` | Overrides for training (inherits default) |
| `configs/inference.yaml` | Inference / API tuning |
| `configs/evaluation.yaml` | Evaluation run settings |

YAML is loaded with `_base_` inheritance. Example:

```yaml
# configs/training.yaml
_base_: "default.yaml"
training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.0001
```

---

## Training

### Command

```bash
python scripts/train.py \
  --config configs/training.yaml \
  --data-dir data \
  --output-dir outputs/training

# Resume
python scripts/train.py --resume checkpoints/best_model.pth

# CLI equivalent
swellsight train --data-dir data --output-dir outputs/training
```

### Outputs

- `checkpoints/best_model.pth` — best validation loss
- `checkpoints/checkpoint_epoch_*.pth` — periodic saves
- `outputs/training/training.log` — log file

### What the trainer expects

- `WaveDataset` in `src/swellsight/data/datasets.py`
- Batches: `input` (4, H, W), `labels`: `height`, `direction`, `breaking_type` (class indices)
- Resolution from `config.data.target_resolution` (default 518×518; multiples of 14)

---

## Inference

### End-to-end (depth + wave metrics)

```bash
python scripts/inference.py \
  --input path/to/beach.jpg \
  --output outputs/inference \
  --checkpoint checkpoints/best_model.pth

# Directory of images
python scripts/inference.py --input path/to/folder/ --output outputs/inference

swellsight analyze --input beach.jpg --checkpoint checkpoints/best_model.pth
```

Writes per-image `*_analysis.json` and `summary.json`.

### Example script (interactive)

```bash
python examples/analyze_beach_cam.py beach.jpg --gpu
```

Uses `WaveAnalysisPipeline` directly (no checkpoint unless you extend `PipelineConfig.wave_checkpoint_path`).

### Load checkpoint in Python

```python
from swellsight.core.pipeline import WaveAnalysisPipeline, PipelineConfig

config = PipelineConfig(
    wave_checkpoint_path="checkpoints/best_model.pth",
    wave_backbone_model="dinov2_vitb14",
)
pipeline = WaveAnalysisPipeline(config=config)
result = pipeline.process_beach_cam_image(rgb_numpy)
```

---

## Evaluation

```bash
python scripts/evaluate.py \
  --model-path checkpoints/best_model.pth \
  --test-data data \
  --output-dir outputs/evaluation

swellsight evaluate --model-path checkpoints/best_model.pth --test-data data
```

Produces `outputs/evaluation/metrics.json` (loss, height MSE, direction/breaking accuracy).

---

## Scripts reference (model-related)

| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/train.py` | **Supported** | Train `WaveAnalysisModel` |
| `scripts/inference.py` | **Supported** | Pipeline inference on images |
| `scripts/evaluate.py` | **Supported** | Metrics on labeled `.npy` data |
| `scripts/check_training_readiness.py` | **Supported** | Environment/data checks |
| `scripts/generate_dummy_data.py` | **Supported** | Smoke-test dataset |
| `scripts/extract_depth_maps.py` | Colab-oriented | Depth-Anything batch |
| `scripts/generate_synthetic_data.py` | Colab-oriented | FLUX synthetic data |
| `scripts/start_api.py` | Partial | REST API (Phase 2) |

Legacy names in old docs (`train_model.py`, `training_config.yaml`) are **removed** — use `train.py` and `configs/training.yaml`.

---

## Known limitations (Phase 1)

1. **Depth + FLUX scripts** still target Google Colab paths; refactor for local runs is planned.
2. **README metrics** (0.18 m MAE, etc.) refer to a completed training run — reproduce with your data and `evaluate.py`.
3. **Breaking classes:** training uses 3 classes; inference heads can be configured for 4 (`NO_BREAKING`) via config when labels support it.
4. **Phase 2** will add user management, surf score API, and web UI — not in this guide.

---

## Quick smoke test (no GPU dataset)

```bash
pip install -e ".[training]"
python scripts/generate_dummy_data.py --output-dir data
python scripts/train.py --data-dir data --output-dir outputs/training
python scripts/evaluate.py --model-path checkpoints/best_model.pth --test-data data
```

---

## Related docs

- [START_HERE.md](START_HERE.md) — First-time orientation
- [TRAINING_FROM_SCRATCH.md](TRAINING_FROM_SCRATCH.md) — Full data pipeline (depth, FLUX, augment)
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) — Additional inference patterns
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common errors
