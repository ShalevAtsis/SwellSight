# SwellSight Training Pipeline Architecture

## Component Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      scripts/train.py                           │
│  (Command-line interface for training)                          │
│                                                                  │
│  • Parses CLI arguments                                         │
│  • Loads configuration via ConfigManager                        │
│  • Initializes WaveAnalysisTrainer                              │
│  • Handles checkpoint resumption                                │
│  • Sets up logging                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              src/swellsight/training/trainer.py                 │
│  (Training orchestration and loop management)                   │
│                                                                  │
│  • Manages training/validation loops                            │
│  • Handles loss calculation (MSE + CrossEntropy)                │
│  • Optimizer management (AdamW/Adam)                            │
│  • Checkpoint saving/loading                                    │
│  • Metrics tracking and logging                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ instantiates
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            src/swellsight/models/wave_model.py                  │
│  (Multi-task wave analysis neural network)                      │
│                                                                  │
│  • DINOv2 backbone (Vision Transformer)                         │
│  • Input adapter (4 channels → 3 channels)                      │
│  • Task heads:                                                  │
│    - Height regression (1 output)                               │
│    - Direction classification (3 classes)                       │
│    - Breaking type classification (3 classes)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Flow

```
configs/training.yaml
        │
        │ loaded by
        ▼
ConfigManager (src/swellsight/utils/config.py)
        │
        │ provides SwellSightConfig
        ▼
┌───────────────────┬───────────────────┬──────────────────┐
│                   │                   │                  │
│  train.py         │  trainer.py       │  wave_model.py   │
│  (CLI setup)      │  (Training loop)  │  (Model arch)    │
│                   │                   │                  │
└───────────────────┴───────────────────┴──────────────────┘
```

## Data Flow During Training

```
DataLoader (train/val)
        │
        │ batch: {'input': Tensor(B,4,H,W), 
        │         'labels': {'height', 'direction', 'breaking_type'}}
        ▼
WaveAnalysisModel.forward()
        │
        ├─► Input Adapter (4→3 channels)
        │
        ├─► DINOv2 Backbone (feature extraction)
        │        │
        │        └─► CLS token (B, embed_dim)
        │
        ├─► Height Head → (B, 1)
        ├─► Direction Head → (B, 3)
        └─► Breaking Head → (B, 3)
        │
        ▼
Loss Calculation (in trainer)
        │
        ├─► MSE Loss (height)
        ├─► CrossEntropy Loss (direction)
        └─► CrossEntropy Loss (breaking)
        │
        ▼
Weighted Sum → Total Loss
        │
        ▼
Backpropagation → Optimizer Step
```

## Key Integration Points

### 1. Config Compatibility
Both dict and SwellSightConfig objects are supported:

```python
# Option A: Dict config
config = {
    'model': {'backbone': 'dinov2_vitb14', ...},
    'training': {'batch_size': 32, ...}
}

# Option B: Config object
config_manager = ConfigManager("configs/training.yaml")
config = config_manager.get_config()

# Both work with:
model = WaveAnalysisModel(config)
trainer = WaveAnalysisTrainer(config)
```

### 2. Import Paths
All imports use relative paths from `swellsight` package:

```python
from swellsight.models.wave_model import WaveAnalysisModel
from swellsight.training.trainer import WaveAnalysisTrainer
from swellsight.utils.config import ConfigManager
```

### 3. Device Management
Automatic GPU/CPU detection:

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(self.device)
```

### 4. Checkpoint Format
Standardized checkpoint structure:

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'metrics': dict,
    'config': dict/object
}
```

## Error Handling Strategy

```
torch.hub.load() fails
        │
        ├─► Log warning
        │
        └─► Use fallback backbone
                │
                └─► Simple CNN → Linear projection
                        │
                        └─► Same interface as DINOv2
```

## Logging Hierarchy

```
swellsight.train (train.py)
    │
    ├─► Training setup logs
    ├─► Configuration validation
    └─► High-level progress
            │
            └─► swellsight.training.trainer (trainer.py)
                    │
                    ├─► Epoch progress
                    ├─► Metrics logging
                    └─► Checkpoint events
                            │
                            └─► swellsight.models.wave_model (wave_model.py)
                                    │
                                    ├─► Model initialization
                                    └─► Backbone loading
```

## Testing Coverage

```
test_integration_wave_model_trainer.py
    │
    ├─► test_model_initialization()
    │   ├─► Dict config ✓
    │   └─► ConfigManager ✓
    │
    ├─► test_model_forward()
    │   └─► Output shapes ✓
    │
    ├─► test_trainer_initialization()
    │   ├─► Dict config ✓
    │   └─► ConfigManager ✓
    │
    ├─► test_training_loop()
    │   ├─► 2 epochs ✓
    │   └─► Checkpoint saving ✓
    │
    └─► test_checkpoint_loading()
        └─► Resume training ✓
```

## Performance Considerations

1. **Backbone Freezing**: Reduces trainable parameters by ~90%
2. **Mixed Precision**: Supported via config (faster training)
3. **Gradient Clipping**: Prevents exploding gradients
4. **Batch Size Adaptation**: Configurable per GPU memory

## Next Integration Steps

1. **Data Pipeline**: Connect WaveDataset to trainer
2. **Metrics Tracking**: Add TensorBoard/WandB integration
3. **Learning Rate Scheduling**: Implement cosine annealing
4. **Early Stopping**: Add validation-based stopping
5. **Multi-GPU**: Add DistributedDataParallel support
