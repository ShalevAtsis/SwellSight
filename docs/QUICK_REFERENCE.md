# SwellSight Quick Reference Card

## Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/SwellSight_Colab.git
cd SwellSight_Colab
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements/base.txt
```

## Basic Usage

### Analyze Single Image

```python
from src.swellsight.core.pipeline import WaveAnalysisPipeline
import cv2

image = cv2.cvtColor(cv2.imread('beach.jpg'), cv2.COLOR_BGR2RGB)
pipeline = WaveAnalysisPipeline()
result = pipeline.process_beach_cam_image(image)

print(f"{result.wave_metrics.height_meters:.1f}m, {result.wave_metrics.direction}")
```

### Batch Processing

```python
images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files]
batch_results = pipeline.process_batch(images)
print(f"Success: {batch_results.get_success_rate():.1%}")
```

### REST API

```bash
# Start server
python -m src.swellsight.api.server

# Analyze image
curl -X POST http://localhost:8000/analyze -F "image=@beach.jpg"
```

## Configuration

### GPU Settings

```python
from src.swellsight.core.pipeline import PipelineConfig

# Use GPU
config = PipelineConfig(use_gpu=True, depth_precision="fp16")

# Force CPU
config = PipelineConfig(use_gpu=False)

# Auto fallback
config = PipelineConfig(use_gpu=True, enable_fallback=True)
```

### Quality Thresholds

```python
config = PipelineConfig(
    confidence_threshold=0.7,        # Min confidence
    depth_quality_threshold=0.5,     # Min depth quality
    prediction_quality_threshold=0.6 # Min prediction quality
)
```

### Performance

```python
config = PipelineConfig(
    enable_optimization=True,        # Enable optimizations
    target_latency_ms=200.0,        # Target latency
    max_processing_time=30.0        # Max processing time
)
```

## Common Tasks

### Check System Status

```python
status = pipeline.get_pipeline_status()
print(f"GPU: {status['hardware_status'].get('gpu_available', False)}")
print(f"Real-time: {status['performance_statistics'].get('real_time_capable', False)}")
```

### Save Results

```python
result = pipeline.process_beach_cam_image(image)
result.save_to_file('results.json')
```

### Monitor Performance

```python
print(f"Processing Time: {result.processing_time:.2f}s")
print(f"Confidence: {result.pipeline_confidence:.1%}")
for stage, time in result.stage_timings.items():
    print(f"  {stage}: {time:.3f}s")
```

### Handle Errors

```python
try:
    result = pipeline.process_beach_cam_image(image)
except Exception as e:
    print(f"Error: {e}")
    # Check warnings
    if result and result.warnings:
        for warning in result.warnings:
            print(f"Warning: {warning}")
```

## Troubleshooting

### GPU Out of Memory
```python
config = PipelineConfig(depth_model_size="base", depth_precision="fp16")
# or
config = PipelineConfig(use_gpu=False)
```

### Slow Processing
```python
config = PipelineConfig(
    enable_optimization=True,
    depth_precision="fp16",
    use_gpu=True
)
```

### Low Confidence
```python
# Check depth quality
if result.depth_quality.overall_score < 0.7:
    print("Low depth quality")
    
# Lower thresholds
config = PipelineConfig(
    confidence_threshold=0.6,
    depth_quality_threshold=0.4
)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze single image |
| `/analyze/batch` | POST | Analyze multiple images |
| `/status` | GET | System status |

## Result Structure

```python
result.wave_metrics.height_meters      # float: Wave height in meters
result.wave_metrics.height_feet        # float: Wave height in feet
result.wave_metrics.direction          # str: "LEFT", "RIGHT", "STRAIGHT"
result.wave_metrics.breaking_type      # str: "SPILLING", "PLUNGING", "SURGING"
result.wave_metrics.height_confidence  # float: 0.0-1.0
result.wave_metrics.extreme_conditions # bool: Extreme wave flag
result.processing_time                 # float: Processing time in seconds
result.pipeline_confidence             # float: Overall confidence 0.0-1.0
result.warnings                        # list: Warning messages
```

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Inference Time | <200ms | ~150ms |
| End-to-End | <30s | ~1-2s |
| Height Accuracy | ±0.2m | ±0.15m |
| Direction Accuracy | 90% | 92% |
| Breaking Accuracy | 92% | 94% |

## Model Sizes

| Size | VRAM | Speed | Accuracy |
|------|------|-------|----------|
| small | 4GB | Fast | Good |
| base | 6GB | Medium | Better |
| large | 8GB+ | Slower | Best |

## Common Patterns

### Production Pipeline
```python
config = PipelineConfig(
    depth_model_size="large",
    depth_precision="fp16",
    use_gpu=True,
    enable_optimization=True,
    confidence_threshold=0.7,
    max_retries=3,
    enable_fallback=True
)
```

### Development/Testing
```python
config = PipelineConfig(
    depth_model_size="base",
    use_gpu=False,
    save_intermediate_results=True,
    output_directory="./debug"
)
```

### High-Throughput Batch
```python
config = PipelineConfig(
    depth_model_size="base",
    depth_precision="fp16",
    use_gpu=True,
    enable_optimization=True,
    save_intermediate_results=False
)
```

## Links

- **Full Guide**: [docs/USER_GUIDE.md](USER_GUIDE.md)
- **API Docs**: [docs/api.md](api.md)
- **Training**: [docs/training.md](training.md)
- **Deployment**: [docs/deployment.md](deployment.md)
