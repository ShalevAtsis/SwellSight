# SwellSight Troubleshooting Guide

Common issues and solutions for the SwellSight Wave Analysis System.

---

## 🔧 Installation Issues

### Issue: "No module named 'src.swellsight'"

**Cause**: Python path not configured correctly

**Solution**:
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in Python
import sys
sys.path.insert(0, '.')
```

### Issue: "CUDA not available" but GPU is present

**Cause**: PyTorch not installed with CUDA support

**Solution**:
```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision

# Install CUDA version (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Issue: "Failed to download model from Hugging Face"

**Cause**: Network issues or authentication required

**Solution**:
```bash
# Login to Hugging Face
huggingface-cli login

# Or set token
export HF_TOKEN="your_token_here"

# Retry download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')"
```

---

## 🧠 Model Loading Issues

### Issue: "RuntimeError: CUDA out of memory"

**Cause**: GPU memory insufficient for model

**Solution 1**: Use CPU
```python
wave_analyzer = DINOv2WaveAnalyzer(device="cpu")
```

**Solution 2**: Use smaller model
```python
depth_extractor = DepthAnythingV2Extractor(model_size="small")  # Instead of "large"
```

**Solution 3**: Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

### Issue: "Model checkpoint not found"

**Cause**: Model not trained or checkpoint path incorrect

**Solution**:
```bash
# Check if checkpoint exists
ls models/checkpoints/best_model.pth

# If not, train model
python scripts/train_model.py

# Or download pre-trained
python scripts/download_pretrained_model.py
```


---

## 📊 Prediction Quality Issues

### Issue: Inaccurate wave height predictions

**Possible Causes**:
1. Poor image quality
2. Insufficient training data
3. Camera angle/distance issues
4. Model not calibrated

**Solutions**:

**Check image quality**:
```python
from src.swellsight.utils.quality_validation import ComprehensiveQualityValidator

validator = ComprehensiveQualityValidator()
is_valid, metrics = validator.validate_input_quality(image, depth_map)

if not is_valid:
    print(f"Quality issues: {metrics}")
```

**Retrain with more data**:
```bash
# Generate more synthetic data
python scripts/generate_synthetic_data.py --num_images 1000

# Retrain model
python scripts/train_model.py
```

**Calibrate confidence scores**:
```python
# Collect calibration data
for image, depth, ground_truth in validation_set:
    wave_metrics, _, _ = analyzer.analyze_waves(image, depth)
    analyzer.add_confidence_calibration_data(wave_metrics, ground_truth)

# Fit calibrators
analyzer.fit_confidence_calibrators(min_samples=50)
analyzer.save_confidence_calibration_data('calibration.pkl')
```

### Issue: Low confidence scores (<60%)

**Cause**: Model uncertain about predictions

**Solutions**:

**Check input quality**:
- Ensure ocean is clearly visible
- Verify depth map quality
- Check for obstructions (people, objects)

**Improve training**:
- Add more diverse training data
- Include similar conditions in training set
- Fine-tune on real beach cam data

**Use quality validation**:
```python
wave_metrics, _, quality_results = analyzer.analyze_waves(image, depth)

if quality_results['input_rejected']:
    print("⚠️ Input quality too low")
    print(f"Issues: {quality_results['input_validation']}")

if quality_results['prediction_validation']:
    if not quality_results['prediction_validation']['is_valid']:
        print("⚠️ Anomalous prediction detected")
```

### Issue: Direction classification always returns "STRAIGHT"

**Cause**: Model not learning directional features

**Solutions**:

**Check training data balance**:
```python
# Verify direction distribution in training data
from collections import Counter
directions = [sample['direction'] for sample in training_dataset]
print(Counter(directions))

# Should be roughly balanced: {'LEFT': ~33%, 'RIGHT': ~33%, 'STRAIGHT': ~33%}
```

**Increase direction loss weight**:
```yaml
# In training config
training:
  loss_weights:
    height: 1.0
    direction: 2.0  # Increase from 1.0
    breaking_type: 1.0
```

**Use adaptive loss weighting**:
```yaml
training:
  adaptive_loss_weighting: true  # Automatically balance losses
```

---

## ⚡ Performance Issues

### Issue: Inference too slow (>30 seconds per image)

**Cause**: Suboptimal configuration or hardware

**Solutions**:

**Enable optimization**:
```python
wave_analyzer = DINOv2WaveAnalyzer(
    enable_optimization=True,  # Enable performance optimizations
    device="cuda"  # Use GPU
)
```

**Use mixed precision**:
```python
depth_extractor = DepthAnythingV2Extractor(
    precision="fp16"  # Use half precision
)
```

**Reduce input resolution**:
```python
# Resize image before processing
import cv2
image_resized = cv2.resize(image, (640, 480))
```

**Check performance metrics**:
```python
wave_metrics, perf_metrics, _ = analyzer.analyze_waves(image, depth)

if perf_metrics:
    print(f"Total time: {perf_metrics.total_time_ms:.1f}ms")
    print(f"Preprocessing: {perf_metrics.preprocessing_time_ms:.1f}ms")
    print(f"Forward pass: {perf_metrics.forward_time_ms:.1f}ms")
    print(f"Postprocessing: {perf_metrics.postprocessing_time_ms:.1f}ms")
```

### Issue: High memory usage

**Cause**: Models not being cleared from memory

**Solutions**:

**Clear GPU cache**:
```python
import torch
torch.cuda.empty_cache()
```

**Use hardware manager**:
```python
from src.swellsight.utils.hardware import HardwareManager

hw_manager = HardwareManager()
hw_manager.cleanup_gpu_memory()
```

**Process in batches**:
```python
# Instead of loading all images at once
for batch in image_batches:
    process_batch(batch)
    torch.cuda.empty_cache()  # Clear after each batch
```

---

## 🎓 Training Issues

### Issue: Training loss not decreasing

**Possible Causes**:
1. Learning rate too high/low
2. Insufficient data
3. Model architecture issues
4. Data quality problems

**Solutions**:

**Adjust learning rate**:
```yaml
# Try different learning rates
training:
  learning_rate: 0.0001  # Reduce if loss exploding
  # or
  learning_rate: 0.01    # Increase if loss plateauing
```

**Use learning rate scheduler**:
```yaml
training:
  scheduler:
    type: "cosine_warmup"
    warmup_epochs: 5
    min_lr: 0.00001
```

**Check data quality**:
```python
# Verify training data
from src.swellsight.data.datasets import WaveDataset

dataset = WaveDataset('data/synthetic')
print(f"Dataset size: {len(dataset)}")

# Check sample
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Image shape: {sample['rgb_image'].shape}")
print(f"Labels: {sample['height_meters']}, {sample['direction_labels']}")
```

**Monitor training metrics**:
```python
# In training loop
print(f"Epoch {epoch}:")
print(f"  Train Loss: {train_loss:.4f}")
print(f"  Val Loss: {val_loss:.4f}")
print(f"  Height MAE: {height_mae:.3f}m")
print(f"  Direction Acc: {direction_acc:.1%}")
print(f"  Breaking Acc: {breaking_acc:.1%}")
```

### Issue: Validation loss increasing (overfitting)

**Cause**: Model memorizing training data

**Solutions**:

**Add regularization**:
```yaml
training:
  weight_decay: 0.001  # Increase from 0.0001
```

**Use early stopping**:
```yaml
training:
  early_stopping_patience: 10  # Stop if no improvement for 10 epochs
```

**Increase training data**:
```bash
# Generate more synthetic data
python scripts/generate_synthetic_data.py --num_images 2000

# Add more augmentations
python scripts/augment_data.py --augmentations_per_image 5
```

**Reduce model complexity**:
```python
# Use smaller backbone
wave_analyzer = DINOv2WaveAnalyzer(
    backbone_model="dinov2_vits14"  # Smaller than vitb14
)
```

### Issue: "NaN loss" during training

**Cause**: Numerical instability

**Solutions**:

**Reduce learning rate**:
```yaml
training:
  learning_rate: 0.0001  # Much smaller
```

**Use gradient clipping**:
```yaml
training:
  gradient_clip_norm: 1.0  # Clip gradients
```

**Check for invalid data**:
```python
# Verify no NaN/Inf in data
import torch
import numpy as np

for batch in train_loader:
    if torch.isnan(batch['rgb_image']).any():
        print("NaN in images!")
    if torch.isinf(batch['height_meters']).any():
        print("Inf in labels!")
```

---

## 🔌 Integration Issues

### Issue: REST API not starting

**Cause**: Port already in use or dependencies missing

**Solutions**:

**Check port availability**:
```bash
# Check if port 8000 is in use
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Use different port
python -m src.swellsight.api.server --port 8001
```

**Install API dependencies**:
```bash
pip install fastapi uvicorn python-multipart
```

### Issue: Notebook kernel crashes

**Cause**: Memory overflow or incompatible dependencies

**Solutions**:

**Restart kernel and clear outputs**:
- Jupyter: Kernel → Restart & Clear Output

**Reduce batch size**:
```python
# In notebook
batch_size = 4  # Reduce from 16
```

**Check memory usage**:
```python
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / (1024 * 1024)
print(f"Memory usage: {memory_mb:.1f}MB")
```

---

## 📞 Getting Help

If you're still experiencing issues:

1. **Check GitHub Issues**: https://github.com/yourusername/SwellSight_Colab/issues
2. **Read Documentation**: https://swellsight.readthedocs.io
3. **Join Discord**: https://discord.gg/swellsight
4. **Create Issue**: Include:
   - Error message (full traceback)
   - System info (OS, Python version, GPU)
   - Steps to reproduce
   - What you've tried

---

## 🔍 Diagnostic Commands

Run these to gather system information for bug reports:

```bash
# System info
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# GPU info
nvidia-smi

# Package versions
pip list | grep -E "torch|transformers|diffusers"

# Disk space
df -h

# Memory
free -h  # Linux
```

Save output and include in bug reports.
