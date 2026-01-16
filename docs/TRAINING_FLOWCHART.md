# SwellSight Training Pipeline - Visual Flowchart

## 🎯 Complete Training Process Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWELLSIGHT TRAINING PIPELINE                 │
│                         (2-3 Days Total)                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: ENVIRONMENT SETUP (15 minutes)                         │
├─────────────────────────────────────────────────────────────────┤
│ • Install Python dependencies                                   │
│ • Setup virtual environment                                     │
│ • Create directory structure                                    │
│                                                                 │
│ Command: pip install -r requirements/training.txt              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: DATA COLLECTION (Variable time)                        │
├─────────────────────────────────────────────────────────────────┤
│ • Gather 100-500 beach cam images                              │
│ • Sources: Surfline, YouTube, local cams                       │
│ • Requirements: 480p-4K, clear ocean visibility                │
│                                                                 │
│ Output: data/raw/beach_cams/*.jpg                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: DATA PREPROCESSING (30 minutes)                        │
├─────────────────────────────────────────────────────────────────┤
│ • Resize images to standard resolution                         │
│ • Normalize and enhance quality                                │
│ • Validate image format                                        │
│                                                                 │
│ Command: python scripts/preprocess_data.py                     │
│ Output: data/processed/beach_cams/*.jpg                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: DEPTH MAP EXTRACTION (1-2 hours)                       │
├─────────────────────────────────────────────────────────────────┤
│ • Use Depth-Anything-V2-Large model                            │
│ • Extract high-sensitivity depth maps                          │
│ • Preserve wave edges and texture                              │
│                                                                 │
│ Command: python scripts/extract_depth_maps.py                  │
│ Output: data/depth_maps/*_depth.npy                            │
│                                                                 │
│ ⚠️  Downloads ~5GB model on first run                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: SYNTHETIC DATA GENERATION (4-8 hours) ⏰               │
├─────────────────────────────────────────────────────────────────┤
│ • Use FLUX.1-dev + ControlNet-Depth                            │
│ • Generate 500+ photorealistic wave images                     │
│ • Automatic labeling (height, direction, breaking type)        │
│ • Balanced across all wave conditions                          │
│                                                                 │
│ Command: python scripts/generate_synthetic_data.py             │
│ Output: data/synthetic/synthetic_*.npy + labels                │
│                                                                 │
│ ⚠️  Downloads ~10GB models on first run                        │
│ ⚠️  This is the LONGEST step - be patient!                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: DATA AUGMENTATION (30 minutes)                         │
├─────────────────────────────────────────────────────────────────┤
│ • Apply weather effects (rain, fog, glare)                     │
│ • Preserve geometric scale for height measurement              │
│ • Create 3x variations per image                               │
│                                                                 │
│ Command: python scripts/augment_data.py                        │
│ Output: data/augmented/*_aug*.npy + labels                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: MODEL TRAINING (8-24 hours) ⏰⏰                        │
├─────────────────────────────────────────────────────────────────┤
│ • Train DINOv2-based multi-task model                          │
│ • 50 epochs with cosine learning rate schedule                 │
│ • Multi-task learning: height + direction + breaking type      │
│ • Automatic checkpointing every 5 epochs                       │
│                                                                 │
│ Command: python scripts/train_model.py                         │
│ Output: models/checkpoints/best_model.pth                      │
│                                                                 │
│ ⚠️  This is the SECOND LONGEST step                            │
│ ⚠️  Monitor training progress in logs                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: MODEL EVALUATION (1 hour)                              │
├─────────────────────────────────────────────────────────────────┤
│ • Test on held-out test set                                    │
│ • Calculate accuracy metrics                                   │
│ • Benchmark inference speed                                    │
│ • Generate evaluation report                                   │
│                                                                 │
│ Command: python scripts/evaluate_model.py                      │
│ Output: Evaluation metrics and performance report              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 9: USE YOUR MODEL! 🎉                                     │
├─────────────────────────────────────────────────────────────────┤
│ • Analyze new beach cam images                                 │
│ • Get wave height, direction, breaking type                    │
│ • Deploy as REST API                                           │
│                                                                 │
│ Command: python examples/analyze_beach_cam.py image.jpg        │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Time Breakdown

```
Setup              ████ 15 min
Data Collection    ████████████████ Variable
Preprocessing      ████ 30 min
Depth Extraction   ████████ 1-2 hours
Synthetic Gen      ████████████████████████████████ 4-8 hours ⏰
Augmentation       ████ 30 min
Training           ████████████████████████████████████████████████ 8-24 hours ⏰⏰
Evaluation         ████████ 1 hour
─────────────────────────────────────────────────────────────────
TOTAL              15-36 hours (excluding data collection)
```

## 🔄 Data Flow Diagram

```
Raw Beach Cam Images (100-500 images)
         │
         ├─→ Preprocessing
         │        │
         │        ↓
         │   Processed Images (640x480)
         │        │
         │        ↓
         │   Depth Extraction (Depth-Anything-V2)
         │        │
         │        ↓
         │   Depth Maps (100-500 depth maps)
         │        │
         │        ↓
         │   Synthetic Generation (FLUX ControlNet)
         │        │
         │        ↓
         │   Synthetic Images + Labels (500+ images)
         │        │
         │        ├─→ Augmentation
         │        │        │
         │        │        ↓
         │        │   Augmented Images (1500+ images)
         │        │        │
         │        └────────┴─→ Combined Dataset
         │                      │
         │                      ↓
         │                 Training Split
         │                      │
         │        ┌─────────────┼─────────────┐
         │        ↓             ↓             ↓
         │    Train (80%)   Val (10%)    Test (10%)
         │        │             │             │
         │        └─────────────┴─────────────┘
         │                      │
         │                      ↓
         │              Model Training
         │                      │
         │                      ↓
         │              Trained Model
         │                      │
         │                      ↓
         └──────────────→  Wave Analysis!
```

## 💾 Disk Space Requirements

```
Component                    Size        Location
─────────────────────────────────────────────────────────────
Raw Images (500 images)      ~2GB        data/raw/
Processed Images             ~1GB        data/processed/
Depth Maps                   ~500MB      data/depth_maps/
Synthetic Images             ~5GB        data/synthetic/
Augmented Images             ~15GB       data/augmented/
Model Checkpoints            ~2GB        models/checkpoints/
Downloaded Models            ~15GB       ~/.cache/huggingface/
─────────────────────────────────────────────────────────────
TOTAL                        ~40GB       (excluding cache)
```

## 🎮 GPU Memory Requirements

```
Task                         VRAM Needed    Recommended GPU
─────────────────────────────────────────────────────────────
Depth Extraction             4-6GB          RTX 2060+
Synthetic Generation         8-10GB         RTX 3070+
Model Training               6-8GB          RTX 2070+
Inference                    2-4GB          RTX 2060+
```

## 🚦 Progress Indicators

### What You'll See During Training

```
Epoch 1/50:
  Train Loss: 2.456 | Val Loss: 2.123
  Height MAE: 0.85m | Direction Acc: 72% | Breaking Acc: 68%
  Time: 12m 34s
  ████████████████████████████████████████ 100% [12:34<00:00]

Epoch 2/50:
  Train Loss: 1.892 | Val Loss: 1.756
  Height MAE: 0.62m | Direction Acc: 79% | Breaking Acc: 75%
  Time: 12m 28s
  ████████████████████████████████████████ 100% [12:28<00:00]

...

Epoch 50/50:
  Train Loss: 0.234 | Val Loss: 0.289
  Height MAE: 0.18m | Direction Acc: 92% | Breaking Acc: 94%
  Time: 12m 31s
  ████████████████████████████████████████ 100% [12:31<00:00]

✓ Training completed!
✓ Best model saved to: models/checkpoints/best_model.pth
```

## 🎯 Expected Results

After training, you should achieve:

```
Metric                       Target      Typical
─────────────────────────────────────────────────
Wave Height MAE              ±0.2m       ±0.15m
Wave Height RMSE             ±0.3m       ±0.25m
Direction Accuracy           90%         92%
Breaking Type Accuracy       92%         94%
Inference Time               <200ms      ~150ms
Overall Confidence           >80%        ~85%
```

## 🔧 Troubleshooting Decision Tree

```
                    Training Started
                          │
                          ↓
                   GPU Available?
                    ╱         ╲
                  Yes          No
                   │            │
                   ↓            ↓
            Use GPU Mode    Use CPU Mode
                   │         (10x slower)
                   │            │
                   ↓            ↓
            Memory Sufficient?
                ╱         ╲
              Yes          No
               │            │
               ↓            ↓
        Train Normally   Reduce Batch Size
               │         (batch_size: 8)
               │            │
               └────────────┘
                      │
                      ↓
              Loss Decreasing?
                ╱         ╲
              Yes          No
               │            │
               ↓            ↓
        Continue      Check Learning Rate
         Training      Verify Data Quality
               │       Check Model Loading
               │            │
               └────────────┘
                      │
                      ↓
              Training Complete!
```

## 📝 Checklist

Use this checklist to track your progress:

```
□ Step 1: Environment Setup
  □ Python 3.8+ installed
  □ Virtual environment created
  □ Dependencies installed
  □ GPU detected (if available)

□ Step 2: Data Collection
  □ 100+ beach cam images collected
  □ Images placed in data/raw/beach_cams/
  □ Images verified (resolution, quality)

□ Step 3: Preprocessing
  □ Preprocessing script run
  □ Processed images in data/processed/
  □ Image count verified

□ Step 4: Depth Extraction
  □ Depth-Anything-V2 model downloaded
  □ Depth maps generated
  □ Depth maps in data/depth_maps/
  □ Visualizations checked

□ Step 5: Synthetic Generation
  □ FLUX models downloaded (~10GB)
  □ 500+ synthetic images generated
  □ Labels created for each image
  □ Dataset balance verified

□ Step 6: Augmentation
  □ Augmentation script run
  □ 3x variations created
  □ Total dataset size: 1500+ images

□ Step 7: Training
  □ Training config created
  □ Training started
  □ Loss decreasing over epochs
  □ Checkpoints being saved
  □ Training completed
  □ Best model saved

□ Step 8: Evaluation
  □ Model evaluated on test set
  □ Metrics meet targets
  □ Performance benchmarked

□ Step 9: Deployment
  □ Model tested on new images
  □ Results look reasonable
  □ Ready for production use
```

## 🎓 Learning Resources

- **[Training from Scratch Guide](TRAINING_FROM_SCRATCH.md)** - Detailed step-by-step
- **[User Guide](USER_GUIDE.md)** - Using the trained model
- **Jupyter Notebooks** - Interactive training pipeline
- **Example Scripts** - Ready-to-run training scripts

---

**Ready to start?** Go to **[Training from Scratch Guide](TRAINING_FROM_SCRATCH.md)**
