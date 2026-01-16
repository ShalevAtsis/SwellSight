# SwellSight Wave Analysis System - User Guide

**Version**: 2.0  
**Last Updated**: January 15, 2026

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Running the Model](#running-the-model)
5. [API Usage](#api-usage)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Advanced Usage](#advanced-usage)

---

## Introduction

SwellSight is an AI-powered wave analysis system that processes beach cam footage to extract critical wave metrics for surfers. This guide will help you get started with running the model for wave analysis.

### What You'll Get

- **Wave Height**: Precise measurements in meters and feet (±0.2m accuracy)
- **Wave Direction**: Classification as Left, Right, or Straight
- **Breaking Type**: Classification as Spilling, Plunging, or Surging
- **Confidence Scores**: Reliability estimates for all predictions
- **Processing Time**: Real-time analysis (<30 seconds end-to-end)

---

## Installation

### System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 16GB RAM
- 50GB free disk space
- GPU with 6GB VRAM (optional but recommended)

**Recommended Requirements:**
- Python 3.10+
- 32GB RAM
- 100GB free SSD storage
- GPU with 12GB+ VRAM (RTX 3080 or better)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/SwellSight_Colab.git
cd SwellSight_Colab
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install base dependencies
pip install -r requirements/base.txt

# For training (optional)
pip install -r requirements/training.txt

# For development (optional)
pip install -r requirements/development.txt
```

### Step 4: Verify Installation

```bash
# Run verification script
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Test imports
python -c "from src.swellsight.core.pipeline import WaveAnalysisPipeline; print('✓ Installation successful')"
```

---

## Quick Start

### Analyze a Single Image

```python
from src.swellsight.core.pipeline import WaveAnalysisPipeline, PipelineConfig
import cv2

# Load beach cam image
image = cv2.imread('path/to/beach_cam_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize pipeline with default configuration
pipeline = WaveAnalysisPipeline()

# Process image
result = pipeline.process_beach_cam_image(image)

# Display results
print(f"Wave Height: {result.wave_metrics.height_meters:.1f}m ({result.wave_metrics.height_feet:.1f}ft)")
print(f"Direction: {result.wave_metrics.direction}")
print(f"Breaking Type: {result.wave_metrics.breaking_type}")
print(f"Confidence: {result.pipeline_confidence:.1%}")
print(f"Processing Time: {result.processing_time:.2f}s")
```

### Expected Output

```
Wave Height: 2.3m (7.5ft)
Direction: LEFT
Breaking Type: SPILLING
Confidence: 87.5%
Processing Time: 1.24s
```

---

## Running the Model

### Method 1: Python Script (Recommended)

Create a file `analyze_waves.py`:

```python
#!/usr/bin/env python3
"""
Simple wave analysis script for beach cam images.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from src.swellsight.core.pipeline import WaveAnalysisPipeline, PipelineConfig

def analyze_image(image_path: str, output_path: str = None):
    """Analyze a single beach cam image."""
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Configure pipeline
    config = PipelineConfig(
        use_gpu=True,  # Set to False if no GPU available
        enable_optimization=True,
        save_intermediate_results=output_path is not None,
        output_directory=output_path
    )
    
    # Initialize pipeline
    print("Initializing wave analysis pipeline...")
    pipeline = WaveAnalysisPipeline(config)
    
    # Process image
    print("Analyzing waves...")
    result = pipeline.process_beach_cam_image(image)
    
    # Display results
    print("\n" + "="*60)
    print("WAVE ANALYSIS RESULTS")
    print("="*60)
    print(f"Wave Height:     {result.wave_metrics.height_meters:.2f}m ({result.wave_metrics.height_feet:.1f}ft)")
    print(f"Direction:       {result.wave_metrics.direction}")
    print(f"Breaking Type:   {result.wave_metrics.breaking_type}")
    print(f"Confidence:      {result.pipeline_confidence:.1%}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    
    if result.wave_metrics.extreme_conditions:
        print("\n⚠️  EXTREME CONDITIONS DETECTED")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    print("="*60)
    
    # Save results if output path provided
    if output_path:
        output_file = Path(output_path) / "analysis_results.json"
        result.save_to_file(str(output_file))
        print(f"\nResults saved to: {output_file}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_waves.py <image_path> [output_directory]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_image(image_path, output_path)
```

Run the script:

```bash
# Analyze single image
python analyze_waves.py beach_cam.jpg

# Analyze and save results
python analyze_waves.py beach_cam.jpg ./output
```

### Method 2: Interactive Python Session

```python
# Start Python interpreter
python

# Import and run
from src.swellsight.core.pipeline import WaveAnalysisPipeline
import cv2

# Load and process
pipeline = WaveAnalysisPipeline()
image = cv2.cvtColor(cv2.imread('beach_cam.jpg'), cv2.COLOR_BGR2RGB)
result = pipeline.process_beach_cam_image(image)

# View results
print(result.wave_metrics)
```

### Method 3: Jupyter Notebook

```python
# In Jupyter notebook cell
%matplotlib inline
import matplotlib.pyplot as plt
from src.swellsight.core.pipeline import WaveAnalysisPipeline
import cv2

# Load image
image = cv2.cvtColor(cv2.imread('beach_cam.jpg'), cv2.COLOR_BGR2RGB)

# Display original
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Beach Cam Image')
plt.axis('off')

# Analyze
pipeline = WaveAnalysisPipeline()
result = pipeline.process_beach_cam_image(image, save_intermediates=True)

# Display depth map
if result.enhanced_depth_map:
    plt.subplot(1, 2, 2)
    plt.imshow(result.enhanced_depth_map.data, cmap='viridis')
    plt.title('Enhanced Depth Map')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Print results
print(f"Height: {result.wave_metrics.height_meters:.1f}m")
print(f"Direction: {result.wave_metrics.direction}")
print(f"Breaking: {result.wave_metrics.breaking_type}")
```

### Method 4: Batch Processing

```python
from src.swellsight.core.pipeline import WaveAnalysisPipeline
import cv2
from pathlib import Path

# Initialize pipeline
pipeline = WaveAnalysisPipeline()

# Load multiple images
image_dir = Path('beach_cam_images')
images = []
for img_path in image_dir.glob('*.jpg'):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

# Process batch
print(f"Processing {len(images)} images...")
batch_results = pipeline.process_batch(images)

# Display summary
print(f"\nBatch Processing Summary:")
print(f"Success Rate: {batch_results.get_success_rate():.1%}")
print(f"Average Time: {batch_results.get_average_processing_time():.2f}s")
print(f"Throughput: {batch_results.batch_statistics['throughput_images_per_second']:.2f} images/sec")

# View individual results
for i, result in enumerate(batch_results.individual_results):
    if result:
        print(f"\nImage {i+1}: {result.wave_metrics.height_meters:.1f}m, {result.wave_metrics.direction}")
```

---

## API Usage

### REST API Server

SwellSight includes a production-ready REST API for remote wave analysis.

#### Starting the API Server

```bash
# Start server with default settings
python -m src.swellsight.api.server

# Start with custom configuration
python -m src.swellsight.api.server --host 0.0.0.0 --port 8000 --workers 4
```

#### API Endpoints

**1. Health Check**

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.0",
  "components": {
    "depth_extractor": true,
    "wave_analyzer": true
  }
}
```

**2. Analyze Single Image**

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@beach_cam.jpg" \
  -F "save_intermediates=false"
```

Response:
```json
{
  "wave_metrics": {
    "height_meters": 2.3,
    "height_feet": 7.5,
    "direction": "LEFT",
    "breaking_type": "SPILLING",
    "height_confidence": 0.89,
    "direction_confidence": 0.85,
    "breaking_confidence": 0.82
  },
  "processing_time": 1.24,
  "pipeline_confidence": 0.853,
  "warnings": []
}
```

**3. Batch Analysis**

```bash
curl -X POST http://localhost:8000/analyze/batch \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg"
```

**4. System Status**

```bash
curl http://localhost:8000/status
```

#### Python API Client

```python
import requests
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

# Analyze image
with open('beach_cam.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post(f"{API_URL}/analyze", files=files)
    result = response.json()
    
print(f"Wave Height: {result['wave_metrics']['height_meters']}m")
print(f"Direction: {result['wave_metrics']['direction']}")
print(f"Breaking Type: {result['wave_metrics']['breaking_type']}")
```

---

## Configuration

### Basic Configuration

```python
from src.swellsight.core.pipeline import PipelineConfig

config = PipelineConfig(
    # Model settings
    depth_model_size="large",      # Options: "small", "base", "large"
    depth_precision="fp16",         # Options: "fp16", "fp32"
    wave_backbone_model="dinov2_vitb14",
    
    # Performance settings
    use_gpu=True,                   # Use GPU if available
    enable_optimization=True,       # Enable performance optimizations
    target_latency_ms=200.0,        # Target inference latency
    
    # Quality thresholds
    confidence_threshold=0.7,       # Minimum confidence for predictions
    depth_quality_threshold=0.5,    # Minimum depth map quality
    
    # Output settings
    save_intermediate_results=False,
    output_directory=None
)

pipeline = WaveAnalysisPipeline(config)
```

### Advanced Configuration

```python
config = PipelineConfig(
    # Depth extraction
    depth_model_size="large",
    depth_precision="fp16",
    depth_enhancement_factor=2.0,   # Depth map enhancement strength
    
    # Wave analysis
    wave_backbone_model="dinov2_vitb14",
    freeze_backbone=True,           # Freeze backbone during inference
    confidence_calibration_method="isotonic",
    
    # Performance
    use_gpu=True,
    enable_optimization=True,
    max_processing_time=30.0,       # Maximum allowed processing time
    target_latency_ms=200.0,
    
    # Quality control
    confidence_threshold=0.7,
    depth_quality_threshold=0.5,
    prediction_quality_threshold=0.6,
    
    # Error handling
    max_retries=3,                  # Retry attempts for failures
    retry_delay=1.0,                # Delay between retries
    enable_fallback=True,           # Enable CPU fallback
    
    # Output
    save_intermediate_results=True,
    output_directory="./output"
)
```

### Configuration File (JSON)

Create `config.json`:

```json
{
  "depth_model_size": "large",
  "depth_precision": "fp16",
  "use_gpu": true,
  "enable_optimization": true,
  "confidence_threshold": 0.7,
  "save_intermediate_results": false
}
```

Load configuration:

```python
import json
from src.swellsight.core.pipeline import PipelineConfig

with open('config.json', 'r') as f:
    config_dict = json.load(f)

config = PipelineConfig(**config_dict)
pipeline = WaveAnalysisPipeline(config)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. GPU Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Option 1: Use smaller model
config = PipelineConfig(
    depth_model_size="base",  # Instead of "large"
    depth_precision="fp16"    # Use FP16 precision
)

# Option 2: Enable CPU fallback
config = PipelineConfig(
    use_gpu=True,
    enable_fallback=True  # Automatically fallback to CPU
)

# Option 3: Force CPU processing
config = PipelineConfig(
    use_gpu=False
)
```

#### 2. Slow Processing

**Issue**: Processing takes longer than expected

**Solutions**:
```python
# Enable all optimizations
config = PipelineConfig(
    enable_optimization=True,
    depth_precision="fp16",  # Faster than fp32
    use_gpu=True
)

# Check system status
pipeline = WaveAnalysisPipeline(config)
status = pipeline.get_pipeline_status()
print(f"GPU Available: {status['hardware_status'].get('gpu_available', False)}")
print(f"Real-time Capable: {status['performance_statistics'].get('real_time_capable', False)}")
```

#### 3. Low Confidence Scores

**Issue**: Predictions have low confidence

**Possible Causes**:
- Poor image quality
- Unusual wave conditions
- Insufficient lighting

**Solutions**:
```python
# Check depth quality
result = pipeline.process_beach_cam_image(image)

if result.depth_quality.overall_score < 0.7:
    print("Low depth quality detected")
    print(f"Edge Preservation: {result.depth_quality.edge_preservation}")
    print(f"Texture Capture: {result.depth_quality.texture_capture}")

# Review warnings
for warning in result.warnings:
    print(f"Warning: {warning}")

# Adjust quality thresholds
config = PipelineConfig(
    depth_quality_threshold=0.4,  # Lower threshold
    confidence_threshold=0.6
)
```

#### 4. Import Errors

**Error**: `ModuleNotFoundError: No module named 'src.swellsight'`

**Solution**:
```bash
# Ensure you're in the project root directory
cd SwellSight_Colab

# Install package in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 5. Model Download Issues

**Error**: `OSError: Can't load model`

**Solution**:
```python
# Pre-download models
from transformers import AutoModel
from huggingface_hub import hf_hub_download

# Download depth model
AutoModel.from_pretrained("depth-anything/Depth-Anything-V2-Large")

# Download DINOv2 model
AutoModel.from_pretrained("facebook/dinov2-base")

print("Models downloaded successfully")
```

---

## Performance Optimization

### GPU Optimization

```python
# Optimal GPU configuration
config = PipelineConfig(
    use_gpu=True,
    enable_optimization=True,
    depth_precision="fp16",  # 2x faster than fp32
    depth_model_size="large"
)

# Monitor GPU usage
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

### Batch Processing Optimization

```python
# Process multiple images efficiently
pipeline = WaveAnalysisPipeline()

# Define progress callback
def progress_callback(current, total, result):
    print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
    if result:
        print(f"  Height: {result.wave_metrics.height_meters:.1f}m")

# Process batch with progress tracking
batch_results = pipeline.process_batch(
    images,
    progress_callback=progress_callback
)

print(f"\nThroughput: {batch_results.batch_statistics['throughput_images_per_second']:.2f} images/sec")
```

### Memory Management

```python
import torch
import gc

# Clear GPU cache between batches
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Process large batches
for batch in image_batches:
    results = pipeline.process_batch(batch)
    # Process results...
    clear_gpu_cache()  # Free memory
```

### Performance Monitoring

```python
# Get detailed performance metrics
pipeline = WaveAnalysisPipeline()

# Process image
result = pipeline.process_beach_cam_image(image)

# View timing breakdown
print("Stage Timings:")
for stage, time_taken in result.stage_timings.items():
    print(f"  {stage}: {time_taken:.3f}s")

# Check system health
health = pipeline.get_system_health_report()
print(f"\nSystem Health: {health['health_status']}")
print(f"Health Score: {health['health_score']:.1%}")

if health['recommendations']:
    print("\nRecommendations:")
    for rec in health['recommendations']:
        print(f"  - {rec}")
```

---

## Advanced Usage

### Streaming Analysis

```python
# Analyze live beach cam stream
def image_generator():
    """Generator that yields beach cam frames."""
    import time
    while True:
        # Fetch latest frame from beach cam
        frame = fetch_beach_cam_frame()
        yield frame
        time.sleep(1)  # 1 FPS

# Process stream
pipeline = WaveAnalysisPipeline()

for result in pipeline.process_streaming(image_generator(), max_images=100):
    if result:
        print(f"Wave: {result.wave_metrics.height_meters:.1f}m, "
              f"{result.wave_metrics.direction}, "
              f"Latency: {result.processing_time*1000:.0f}ms")
```

### Custom Post-Processing

```python
# Add custom analysis
def analyze_wave_trends(results_list):
    """Analyze wave height trends over time."""
    heights = [r.wave_metrics.height_meters for r in results_list]
    
    avg_height = np.mean(heights)
    max_height = np.max(heights)
    trend = "increasing" if heights[-1] > heights[0] else "decreasing"
    
    return {
        "average_height": avg_height,
        "max_height": max_height,
        "trend": trend
    }

# Collect results over time
results = []
for image in images:
    result = pipeline.process_beach_cam_image(image)
    results.append(result)

# Analyze trends
trends = analyze_wave_trends(results)
print(f"Average Height: {trends['average_height']:.1f}m")
print(f"Max Height: {trends['max_height']:.1f}m")
print(f"Trend: {trends['trend']}")
```

### Integration with Monitoring Systems

```python
from src.swellsight.utils.monitoring import system_monitor

# Start system monitoring
system_monitor.start_monitoring(interval_seconds=60)

# Run pipeline
pipeline = WaveAnalysisPipeline()
result = pipeline.process_beach_cam_image(image)

# Check for alerts
alerts = system_monitor.get_active_alerts()
for alert in alerts:
    print(f"Alert: {alert.title}")
    print(f"Severity: {alert.severity.value}")
    print(f"Message: {alert.message}")

# Get system status
status = system_monitor.get_system_status()
print(f"Overall Health: {status['overall_health']}")

# Stop monitoring
system_monitor.stop_monitoring()
```

### Saving and Loading Results

```python
# Save results to file
result = pipeline.process_beach_cam_image(image)
result.save_to_file('wave_analysis_results.json')

# Load and analyze later
import json
with open('wave_analysis_results.json', 'r') as f:
    saved_results = json.load(f)

print(f"Saved Height: {saved_results['wave_metrics']['height_meters']}m")
```

---

## Best Practices

### 1. Image Quality

- Use images with resolution between 480p and 4K
- Ensure ocean is visible in the frame
- Avoid heavily compressed images
- Prefer images with good lighting conditions

### 2. Performance

- Use GPU when available for 5-10x speedup
- Enable FP16 precision for 2x speedup with minimal accuracy loss
- Process images in batches for better throughput
- Clear GPU cache between large batches

### 3. Reliability

- Check confidence scores before using predictions
- Review warnings for potential issues
- Monitor system health for production deployments
- Implement retry logic for critical applications

### 4. Production Deployment

- Use the REST API for remote access
- Enable monitoring and alerting
- Set appropriate quality thresholds
- Implement result caching for frequently analyzed locations

---

## Getting Help

### Documentation

- **API Reference**: See `docs/api.md`
- **Training Guide**: See `docs/training.md`
- **Deployment Guide**: See `docs/deployment.md`
- **Integration Guide**: See `INTEGRATION_GUIDE.md`

### Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/SwellSight_Colab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/SwellSight_Colab/discussions)
- **Email**: support@swellsight.ai

### Community

- Join our Discord server for real-time help
- Follow development updates on Twitter
- Check out example notebooks in `notebooks/`

---

## Next Steps

1. **Try the Quick Start** examples above
2. **Explore the Jupyter Notebooks** for interactive analysis
3. **Read the API Documentation** for advanced features
4. **Join the Community** to share your results

Happy wave analyzing! 🏄‍♂️🌊
