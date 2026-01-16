# Training SwellSight from Scratch - Complete Guide

**Status**: You are here because you haven't trained the model yet  
**Goal**: Train a complete wave analysis model from raw beach cam images  
**Time Required**: 2-3 days (depending on hardware and dataset size)

---

## 📋 Overview

This guide walks you through the complete training pipeline:

1. **Data Collection** - Gather beach cam images
2. **Data Preprocessing** - Prepare images for training
3. **Depth Extraction** - Generate depth maps from images
4. **Synthetic Data Generation** - Create labeled training data
5. **Data Augmentation** - Expand dataset with variations
6. **Model Training** - Train the multi-task wave analyzer
7. **Model Evaluation** - Validate performance
8. **Model Deployment** - Use your trained model

---

## 🎯 Prerequisites

### Hardware Requirements

**Minimum** (Training will be slow):
- 16GB RAM
- GPU with 8GB VRAM (RTX 2070 or better)
- 100GB free disk space
- Good internet connection (for downloading models)

**Recommended** (Faster training):
- 32GB+ RAM
- GPU with 12GB+ VRAM (RTX 3080 or better)
- 200GB+ SSD storage
- Fast internet connection

### Software Requirements

```bash
# Check Python version (need 3.8+)
python --version

# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check GPU
nvidia-smi  # Should show your GPU
```

---

## Step 1: Environment Setup (15 minutes)

### 1.0 Check Training Readiness (Optional but Recommended)

Before starting, run our readiness checker to verify your system:

```bash
python scripts/check_training_readiness.py
```

This will check:
- Python version
- GPU availability
- Disk space
- Required packages
- Directory structure
- Data availability

**Expected Output:**
```
╔════════════════════════════════════════════════════════════════════╗
║               SwellSight Training Readiness Check                  ║
╚════════════════════════════════════════════════════════════════════╝

==================================================================
  System Requirements
==================================================================
✓ Python Version.......................................... PASS
  → Found Python 3.10.8
✓ GPU Availability........................................ PASS
  → NVIDIA GeForce RTX 3080 with 10.0GB VRAM
✓ Disk Space.............................................. PASS
  → 156.3GB free

==================================================================
  Software Dependencies
==================================================================
✓ Required Packages....................................... PASS
  → All 8 required packages installed

⚠️  System is ready, but data preparation needed!

Next steps:
  1. Collect beach cam images → data/raw/beach_cams/
  2. Extract depth maps: python scripts/extract_depth_maps.py
  3. Generate synthetic data: python scripts/generate_synthetic_data.py
```

### 1.1 Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/SwellSight_Colab.git
cd SwellSight_Colab

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 1.2 Install Dependencies

```bash
# Install base dependencies
pip install -r requirements/base.txt

# Install training dependencies (includes all ML libraries)
pip install -r requirements/training.txt

# Verify installation
python -c "import torch; import transformers; print('✓ Installation successful')"
```

### 1.3 Create Directory Structure

```bash
# Create data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/synthetic
mkdir -p data/depth_maps
mkdir -p data/augmented

# Create model directories
mkdir -p models/checkpoints
mkdir -p models/final

# Create output directories
mkdir -p outputs/logs
mkdir -p outputs/evaluation
mkdir -p outputs/visualizations
```

---

## Step 2: Data Collection (Variable time)

You need beach cam images showing waves. Here are your options:

### Option A: Use Existing Beach Cam Footage

**Recommended Sources:**
- Surfline.com beach cams (screenshot or download)
- Local beach webcams
- YouTube surf videos (extract frames)
- Your own beach photos

**Requirements:**
- Minimum 100 images (more is better)
- Resolution: 480p to 4K
- Clear ocean visibility
- Variety of conditions (different wave heights, lighting, weather)

### Option B: Download Sample Dataset

```bash
# Download sample beach cam dataset (if available)
# Replace with actual dataset URL
wget https://example.com/beach_cam_dataset.zip
unzip beach_cam_dataset.zip -d data/raw/
```

### Option C: Use the Provided Notebooks

The project includes notebooks that can help you collect data:

```bash
# Open Jupyter
jupyter notebook

# Navigate to:
# 02_Data_Import_and_Preprocessing.ipynb
```

### 2.1 Organize Your Data

```bash
# Place your beach cam images in:
data/raw/beach_cams/
  ├── image_001.jpg
  ├── image_002.jpg
  ├── image_003.jpg
  └── ...

# Recommended: At least 100-500 images for good results
```

### 2.2 Verify Data

```python
# Run this to check your data
from pathlib import Path
import cv2

data_dir = Path('data/raw/beach_cams')
images = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))

print(f"Found {len(images)} images")

# Check first image
if images:
    img = cv2.imread(str(images[0]))
    print(f"Sample image shape: {img.shape}")
    print(f"Resolution: {img.shape[1]}x{img.shape[0]}")
```

---

## Step 3: Data Preprocessing (30 minutes)

### 3.1 Run Preprocessing Notebook

```bash
# Open Jupyter
jupyter notebook

# Open and run:
# 02_Data_Import_and_Preprocessing.ipynb
```

**Or use the preprocessing script:**

```python
# Create preprocessing script: scripts/preprocess_data.py
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, target_size=(640, 480)):
    """Preprocess beach cam images."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    print(f"Preprocessing {len(images)} images...")
    
    for img_path in tqdm(images):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Resize to target size
        img_resized = cv2.resize(img, target_size)
        
        # Save preprocessed image
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), img_resized, 
                   [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"✓ Preprocessed {len(images)} images")

if __name__ == "__main__":
    preprocess_images('data/raw/beach_cams', 'data/processed/beach_cams')
```

Run it:
```bash
python scripts/preprocess_data.py
```

---

## Step 4: Depth Map Extraction (1-2 hours)

This step generates depth maps from your beach cam images using Depth-Anything-V2.

### 4.1 Run Depth Extraction Notebook

```bash
jupyter notebook

# Open and run:
# 03_Depth_Anything_V2_Extraction.ipynb
```

**Or use the depth extraction script:**

```python
# Create script: scripts/extract_depth_maps.py
import sys
sys.path.insert(0, '.')

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from src.swellsight.core.depth_extractor import DepthAnythingV2Extractor

def extract_depth_maps(input_dir, output_dir, use_gpu=True):
    """Extract depth maps from beach cam images."""
    
    print("Initializing Depth-Anything-V2...")
    extractor = DepthAnythingV2Extractor(
        model_size="large",
        precision="fp16" if use_gpu else "fp32",
        enable_optimization=True
    )
    print("✓ Depth extractor initialized")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    print(f"\nExtracting depth maps for {len(images)} images...")
    
    for img_path in tqdm(images):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract depth
        depth_map = extractor.extract_depth(img_rgb)
        
        # Handle tuple return (depth_map, performance_metrics)
        if isinstance(depth_map, tuple):
            depth_map = depth_map[0]
        
        # Save depth map
        output_file = output_path / f"{img_path.stem}_depth.npy"
        np.save(output_file, depth_map.data)
        
        # Also save visualization
        depth_vis = (depth_map.data * 255).astype(np.uint8)
        vis_file = output_path / f"{img_path.stem}_depth_vis.jpg"
        cv2.imwrite(str(vis_file), depth_vis)
    
    print(f"✓ Extracted {len(images)} depth maps")

if __name__ == "__main__":
    extract_depth_maps(
        'data/processed/beach_cams',
        'data/depth_maps',
        use_gpu=True
    )
```

Run it:
```bash
python scripts/extract_depth_maps.py
```

**Expected Output:**
```
Initializing Depth-Anything-V2...
✓ Depth extractor initialized

Extracting depth maps for 100 images...
100%|████████████████████| 100/100 [02:15<00:00,  1.35s/it]
✓ Extracted 100 depth maps
```

---

## Step 5: Synthetic Data Generation (4-8 hours)

This is the key step! Generate labeled synthetic wave images using FLUX ControlNet.

### 5.1 Run Synthetic Generation Notebook

```bash
jupyter notebook

# Open and run:
# 05_FLUX_ControlNet_Synthetic_Generation_Enhanced.ipynb
```

**Or use the synthetic generation script:**

```python
# Create script: scripts/generate_synthetic_data.py
import sys
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.swellsight.core.synthetic_generator import (
    FLUXControlNetGenerator, 
    WeatherConditions,
    GenerationConfig
)
from src.swellsight.core.depth_extractor import DepthMap

def generate_synthetic_dataset(depth_dir, output_dir, num_images=500):
    """Generate synthetic wave images from depth maps."""
    
    print("Initializing FLUX ControlNet Generator...")
    print("⚠️  This will download large models (~10GB) on first run")
    
    generator = FLUXControlNetGenerator()
    print("✓ Generator initialized")
    
    depth_path = Path(depth_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load depth maps
    depth_files = list(depth_path.glob('*_depth.npy'))
    print(f"\nFound {len(depth_files)} depth maps")
    
    if len(depth_files) == 0:
        print("❌ No depth maps found! Run depth extraction first.")
        return
    
    # Generate balanced dataset
    print(f"\nGenerating {num_images} synthetic images...")
    print("This will take several hours depending on your GPU...")
    
    # Use the built-in balanced dataset generation
    labeled_dataset = generator.create_balanced_dataset(target_size=num_images)
    
    # Save synthetic images and labels
    print("\nSaving synthetic dataset...")
    for i, synthetic_image in enumerate(tqdm(labeled_dataset.images)):
        # Save RGB image
        img_file = output_path / f"synthetic_{i:04d}.npy"
        np.save(img_file, synthetic_image.rgb_data)
        
        # Save labels
        label_file = output_path / f"synthetic_{i:04d}_labels.npy"
        labels = {
            'height_meters': synthetic_image.ground_truth_labels.height_meters,
            'direction': synthetic_image.ground_truth_labels.direction,
            'breaking_type': synthetic_image.ground_truth_labels.breaking_type,
            'height_confidence': synthetic_image.ground_truth_labels.height_confidence,
            'direction_confidence': synthetic_image.ground_truth_labels.direction_confidence,
            'breaking_confidence': synthetic_image.ground_truth_labels.breaking_confidence
        }
        np.save(label_file, labels)
    
    print(f"\n✓ Generated {len(labeled_dataset.images)} synthetic images")
    print(f"✓ Dataset statistics:")
    print(f"   Average height: {labeled_dataset.statistics['height_statistics']['mean']:.2f}m")
    print(f"   Height range: {labeled_dataset.statistics['height_statistics']['min']:.2f}m - {labeled_dataset.statistics['height_statistics']['max']:.2f}m")

if __name__ == "__main__":
    generate_synthetic_dataset(
        'data/depth_maps',
        'data/synthetic',
        num_images=500  # Adjust based on your needs
    )
```

Run it:
```bash
# This will take several hours!
python scripts/generate_synthetic_data.py
```

**Expected Output:**
```
Initializing FLUX ControlNet Generator...
⚠️  This will download large models (~10GB) on first run
Downloading models... (this may take 10-30 minutes)
✓ Generator initialized

Found 100 depth maps

Generating 500 synthetic images...
This will take several hours depending on your GPU...
Progress: 10% (50/500) [Est. 3h 20m remaining]
...
✓ Generated 500 synthetic images
✓ Dataset statistics:
   Average height: 2.8m
   Height range: 0.6m - 7.2m
```

---

## Step 6: Data Augmentation (30 minutes)

Expand your dataset with augmentations while preserving wave scale.

### 6.1 Run Augmentation Notebook

```bash
jupyter notebook

# Open and run:
# 04_Data_Augmentation_System_Enhanced.ipynb
```

**Or use the augmentation script:**

```python
# Create script: scripts/augment_data.py
import sys
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.swellsight.data.augmentation import WaveAugmentation

def augment_dataset(input_dir, output_dir, augmentations_per_image=3):
    """Apply augmentations to synthetic dataset."""
    
    print("Initializing augmentation system...")
    augmenter = WaveAugmentation(preserve_scale=True)
    print("✓ Augmenter initialized")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load synthetic images
    image_files = list(input_path.glob('synthetic_*.npy'))
    image_files = [f for f in image_files if '_labels' not in f.name and '_depth' not in f.name]
    
    print(f"\nAugmenting {len(image_files)} images...")
    print(f"Creating {augmentations_per_image} variations per image")
    
    total_generated = 0
    
    for img_file in tqdm(image_files):
        # Load image and labels
        img = np.load(img_file)
        label_file = img_file.parent / f"{img_file.stem}_labels.npy"
        labels = np.load(label_file, allow_pickle=True).item()
        
        # Generate augmentations
        for aug_idx in range(augmentations_per_image):
            # Apply augmentation
            aug_img = augmenter.augment(img)
            
            # Save augmented image
            aug_file = output_path / f"{img_file.stem}_aug{aug_idx}.npy"
            np.save(aug_file, aug_img)
            
            # Copy labels (augmentation preserves wave properties)
            aug_label_file = output_path / f"{img_file.stem}_aug{aug_idx}_labels.npy"
            np.save(aug_label_file, labels)
            
            total_generated += 1
    
    print(f"\n✓ Generated {total_generated} augmented images")
    print(f"✓ Total dataset size: {len(image_files) + total_generated} images")

if __name__ == "__main__":
    augment_dataset(
        'data/synthetic',
        'data/augmented',
        augmentations_per_image=3
    )
```

Run it:
```bash
python scripts/augment_data.py
```

---

## Step 7: Model Training (8-24 hours)

Now train the multi-task wave analyzer!

### 7.1 Prepare Training Configuration

Create `configs/training_config.yaml`:

```yaml
# Training Configuration
model:
  backbone: "dinov2_vitb14"
  freeze_backbone: true
  input_channels: 4  # RGB + Depth
  
training:
  batch_size: 16  # Adjust based on GPU memory
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  
  # Multi-task loss weights
  loss_weights:
    height: 1.0
    direction: 1.0
    breaking_type: 1.0
  
  # Learning rate schedule
  scheduler:
    type: "cosine"
    warmup_epochs: 5
    min_lr: 0.00001

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  
  synthetic_data_dir: "data/synthetic"
  augmented_data_dir: "data/augmented"
  real_data_dir: "data/processed/beach_cams"  # If you have labeled real data
  
hardware:
  use_gpu: true
  mixed_precision: true  # FP16 training for speed
  num_workers: 4

checkpointing:
  save_dir: "models/checkpoints"
  save_frequency: 5  # Save every 5 epochs
  keep_best: true
```

### 7.2 Run Training Notebook

```bash
jupyter notebook

# Open and run:
# 06_Model_Training_Pipeline.ipynb
```

**Or use the training script:**

```python
# Create script: scripts/train_model.py
import sys
sys.path.insert(0, '.')

import yaml
from pathlib import Path
import torch
from src.swellsight.training.trainer import WaveAnalysisTrainer
from src.swellsight.data.datasets import WaveDataset
from torch.utils.data import DataLoader

def train_model(config_path='configs/training_config.yaml'):
    """Train the wave analysis model."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("SwellSight Model Training")
    print("="*60)
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = WaveDataset(
        data_dir=config['data']['synthetic_data_dir'],
        split='train',
        train_ratio=config['data']['train_split']
    )
    
    val_dataset = WaveDataset(
        data_dir=config['data']['synthetic_data_dir'],
        split='val',
        train_ratio=config['data']['train_split']
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers']
    )
    
    # Initialize trainer
    print("\n🔧 Initializing trainer...")
    trainer = WaveAnalysisTrainer(config)
    print("✓ Trainer initialized")
    
    # Train model
    print("\n🚀 Starting training...")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"GPU: {torch.cuda.is_available()}")
    print()
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {config['checkpointing']['save_dir']}/best_model.pth")

if __name__ == "__main__":
    train_model()
```

Run it:
```bash
# This will take 8-24 hours depending on GPU!
python scripts/train_model.py
```

**Expected Output:**
```
============================================================
SwellSight Model Training
============================================================

📊 Loading datasets...
✓ Train samples: 1600
✓ Val samples: 200

🔧 Initializing trainer...
✓ Trainer initialized

🚀 Starting training...
Epochs: 50
Batch size: 16
Learning rate: 0.001
GPU: True

Epoch 1/50:
  Train Loss: 2.456 | Val Loss: 2.123
  Height MAE: 0.85m | Direction Acc: 72% | Breaking Acc: 68%
  Time: 12m 34s

Epoch 2/50:
  Train Loss: 1.892 | Val Loss: 1.756
  Height MAE: 0.62m | Direction Acc: 79% | Breaking Acc: 75%
  Time: 12m 28s

...

Epoch 50/50:
  Train Loss: 0.234 | Val Loss: 0.289
  Height MAE: 0.18m | Direction Acc: 92% | Breaking Acc: 94%
  Time: 12m 31s

✓ Training completed!
✓ Best model saved to: models/checkpoints/best_model.pth
```

---

## Step 8: Model Evaluation (1 hour)

Evaluate your trained model's performance.

### 8.1 Run Evaluation Notebook

```bash
jupyter notebook

# Open and run:
# 08_Model_Evaluation_and_Validation.ipynb
```

**Or use the evaluation script:**

```python
# Create script: scripts/evaluate_model.py
import sys
sys.path.insert(0, '.')

from pathlib import Path
import torch
from src.swellsight.evaluation.evaluator import ModelEvaluator
from src.swellsight.models.wave_analyzer import DINOv2WaveAnalyzer
from src.swellsight.data.datasets import WaveDataset
from torch.utils.data import DataLoader

def evaluate_model(model_path='models/checkpoints/best_model.pth'):
    """Evaluate the trained model."""
    
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    model = DINOv2WaveAnalyzer()
    model.load_state_dict(torch.load(model_path))
    print("✓ Model loaded")
    
    # Create test dataset
    print("\n📊 Loading test dataset...")
    test_dataset = WaveDataset(
        data_dir='data/synthetic',
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model)
    
    # Evaluate accuracy
    print("\n🔍 Evaluating accuracy...")
    accuracy_metrics = evaluator.evaluate_accuracy(test_loader)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\n📏 Wave Height:")
    print(f"   MAE: {accuracy_metrics.height_metrics.mae:.3f}m")
    print(f"   RMSE: {accuracy_metrics.height_metrics.rmse:.3f}m")
    print(f"   Within ±0.2m: {accuracy_metrics.height_metrics.accuracy_within_02m:.1%}")
    
    print(f"\n🧭 Direction:")
    print(f"   Accuracy: {accuracy_metrics.direction_metrics.accuracy:.1%}")
    print(f"   F1-Score: {accuracy_metrics.direction_metrics.macro_avg_f1:.3f}")
    
    print(f"\n💥 Breaking Type:")
    print(f"   Accuracy: {accuracy_metrics.breaking_type_metrics.accuracy:.1%}")
    print(f"   F1-Score: {accuracy_metrics.breaking_type_metrics.macro_avg_f1:.3f}")
    
    print(f"\n⭐ Overall Score: {accuracy_metrics.overall_score:.1%}")
    print("="*60)
    
    # Benchmark performance
    print("\n⚡ Benchmarking performance...")
    perf_metrics = evaluator.benchmark_performance(test_loader)
    
    print(f"\n   Inference Time: {perf_metrics['inference_time_ms']:.1f}ms")
    print(f"   Throughput: {perf_metrics['throughput_images_per_second']:.2f} images/sec")
    print(f"   Memory Usage: {perf_metrics['memory_usage_mb']:.1f}MB")

if __name__ == "__main__":
    evaluate_model()
```

Run it:
```bash
python scripts/evaluate_model.py
```

---

## Step 9: Use Your Trained Model (5 minutes)

Now you can use your trained model for wave analysis!

```python
# Use your trained model
from src.swellsight.core.pipeline import WaveAnalysisPipeline, PipelineConfig
import cv2

# Configure to use your trained model
config = PipelineConfig(
    wave_model_path='models/checkpoints/best_model.pth',  # Your trained model
    use_gpu=True
)

# Initialize pipeline
pipeline = WaveAnalysisPipeline(config)

# Analyze a new beach cam image
image = cv2.imread('new_beach_cam.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = pipeline.process_beach_cam_image(image)

print(f"Wave Height: {result.wave_metrics.height_meters:.1f}m")
print(f"Direction: {result.wave_metrics.direction}")
print(f"Breaking Type: {result.wave_metrics.breaking_type}")
```

---

## 📊 Training Timeline Summary

| Step | Time | Description |
|------|------|-------------|
| 1. Setup | 15 min | Install dependencies |
| 2. Data Collection | Variable | Gather beach cam images |
| 3. Preprocessing | 30 min | Prepare images |
| 4. Depth Extraction | 1-2 hours | Generate depth maps |
| 5. Synthetic Generation | 4-8 hours | Create labeled data |
| 6. Augmentation | 30 min | Expand dataset |
| 7. Model Training | 8-24 hours | Train wave analyzer |
| 8. Evaluation | 1 hour | Validate performance |
| **Total** | **15-36 hours** | **Complete pipeline** |

---

## 🎯 Quick Start Commands

```bash
# 0. Check if you're ready
python scripts/check_training_readiness.py

# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/training.txt

# 2. Preprocess data
python scripts/preprocess_data.py

# 3. Extract depth maps
python scripts/extract_depth_maps.py

# 4. Generate synthetic data (LONG!)
python scripts/generate_synthetic_data.py

# 5. Augment data
python scripts/augment_data.py

# 6. Train model (VERY LONG!)
python scripts/train_model.py

# 7. Evaluate model
python scripts/evaluate_model.py

# 8. Use model
python examples/analyze_beach_cam.py new_image.jpg
```

---

## 🆘 Troubleshooting

### "CUDA out of memory" during training

**Solution**: Reduce batch size in `configs/training_config.yaml`:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### "No depth maps found"

**Solution**: Make sure Step 4 completed successfully:
```bash
ls data/depth_maps/*.npy  # Should show depth map files
```

### Synthetic generation is too slow

**Solution**: Generate fewer images initially:
```python
# In generate_synthetic_data.py, change:
num_images=100  # Instead of 500
```

### Training loss not decreasing

**Possible causes**:
- Learning rate too high/low
- Insufficient data
- Model not loading properly

**Solution**: Check training logs and adjust learning rate.

---

## 📚 Next Steps

After training:

1. **Fine-tune on real data** (if you have labeled real beach cam images)
2. **Deploy your model** - See [Deployment Guide](deployment.md)
3. **Create REST API** - See [API Documentation](api.md)
4. **Monitor performance** - Track metrics over time

---

## 💡 Tips for Success

1. **Start Small**: Train on 100 synthetic images first to verify pipeline works
2. **Monitor Training**: Watch loss curves to ensure model is learning
3. **Save Checkpoints**: Training can be interrupted, checkpoints let you resume
4. **Use GPU**: Training on CPU will take 10x longer
5. **Validate Early**: Test on a few images before full evaluation

---

**Need Help?** 
- Check [User Guide](USER_GUIDE.md) for usage after training
- See [GitHub Issues](https://github.com/yourusername/SwellSight_Colab/issues) for problems
- Join [Discord](https://discord.gg/swellsight) for community support

Good luck with your training! 🏄‍♂️🌊
