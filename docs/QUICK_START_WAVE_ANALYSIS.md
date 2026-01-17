# Quick Start Guide: Wave Analysis with SwellSight

**Goal**: Get wave metrics from beach cam images in under 5 minutes  
**Prerequisites**: Trained model available (see [TRAINING_FROM_SCRATCH.md](TRAINING_FROM_SCRATCH.md) if not)

---

## 🚀 Installation (2 minutes)

```bash
# Clone and setup
git clone https://github.com/yourusername/SwellSight_Colab.git
cd SwellSight_Colab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt

# Verify installation
python -c "from src.swellsight.core.wave_analyzer import DINOv2WaveAnalyzer; print('✓ Ready')"
```

---

## 📸 Analyze a Single Image (1 minute)

### Method 1: Python Script (Recommended)

```python
from src.swellsight.core.wave_analyzer import DINOv2WaveAnalyzer
from src.swellsight.core.depth_extractor import DepthAnythingV2Extractor
import cv2

# Load image
image = cv2.imread('beach_cam.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize components
depth_extractor = DepthAnythingV2Extractor(model_size="large")
wave_analyzer = DINOv2WaveAnalyzer()

# Extract depth and analyze
depth_map, _ = depth_extractor.extract_depth(image_rgb)
wave_metrics, _, _ = wave_analyzer.analyze_waves(image_rgb, depth_map)

# Display results
print(f"🌊 Wave Height: {wave_metrics.height_meters:.1f}m ({wave_metrics.height_feet:.1f}ft)")
print(f"🧭 Direction: {wave_metrics.direction}")
print(f"💥 Breaking Type: {wave_metrics.breaking_type}")
print(f"⭐ Confidence: {wave_metrics.height_confidence:.0%}")
```


### Method 2: Command Line

```bash
# Use the example script
python examples/analyze_beach_cam.py beach_cam.jpg

# With GPU acceleration
python examples/analyze_beach_cam.py beach_cam.jpg --gpu

# Save results to file
python examples/analyze_beach_cam.py beach_cam.jpg --output results.json
```

---

## 📁 Batch Processing (2 minutes)

Process multiple images at once:

```python
from pathlib import Path
from src.swellsight.core.wave_analyzer import DINOv2WaveAnalyzer
from src.swellsight.core.depth_extractor import DepthAnythingV2Extractor
import cv2
import json

# Initialize once
depth_extractor = DepthAnythingV2Extractor(model_size="large")
wave_analyzer = DINOv2WaveAnalyzer(enable_optimization=True)

# Process directory
image_dir = Path('beach_cams')
results = []

for img_path in image_dir.glob('*.jpg'):
    # Load and process
    image = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    depth_map, _ = depth_extractor.extract_depth(image_rgb)
    wave_metrics, _, _ = wave_analyzer.analyze_waves(image_rgb, depth_map)
    
    # Store results
    results.append({
        'filename': img_path.name,
        'height_meters': wave_metrics.height_meters,
        'direction': wave_metrics.direction,
        'breaking_type': wave_metrics.breaking_type,
        'confidence': wave_metrics.height_confidence
    })

# Save results
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Processed {len(results)} images")
```

---

## 🎥 Real-Time Video Analysis

Process video stream or file:

```python
from src.swellsight.core.wave_analyzer import DINOv2WaveAnalyzer
from src.swellsight.core.depth_extractor import DepthAnythingV2Extractor
import cv2

# Initialize
depth_extractor = DepthAnythingV2Extractor(model_size="large")
wave_analyzer = DINOv2WaveAnalyzer(enable_optimization=True)

# Open video
cap = cv2.VideoCapture('beach_cam_video.mp4')  # or 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map, _ = depth_extractor.extract_depth(frame_rgb)
    wave_metrics, _, _ = wave_analyzer.analyze_waves(frame_rgb, depth_map)
    
    # Overlay results
    cv2.putText(frame, f"Height: {wave_metrics.height_meters:.1f}m", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Direction: {wave_metrics.direction}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Wave Analysis', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🔍 Understanding Results

### Wave Height
- **Range**: 0.5m - 8.0m (typical surfable range)
- **Accuracy**: ±0.2m with trained model
- **Extreme Conditions**: Flagged if <0.5m or >8.0m

### Direction
- **LEFT**: Wave breaks from right to left (surfer's perspective)
- **RIGHT**: Wave breaks from left to right
- **STRAIGHT**: Wave breaks straight toward shore (closeout)

### Breaking Type
- **SPILLING**: Gentle, foamy break (beginner-friendly)
- **PLUNGING**: Hollow, barreling wave (advanced)
- **SURGING**: Wave breaks directly on shore (not surfable)
- **NO_BREAKING**: No clear breaking pattern detected

### Confidence Scores
- **>80%**: High confidence, reliable prediction
- **60-80%**: Moderate confidence, generally reliable
- **<60%**: Low confidence, verify visually

---

## 🆘 Troubleshooting

### "Model not found" error
```bash
# Download pre-trained model
python scripts/download_pretrained_model.py

# Or train your own
python scripts/train_model.py
```

### "CUDA out of memory"
```python
# Use CPU instead
wave_analyzer = DINOv2WaveAnalyzer(device="cpu")

# Or reduce batch size for batch processing
```

### Poor predictions
- Check image quality (resolution, clarity)
- Ensure ocean is visible in image
- Verify depth map quality
- Consider retraining with more data

---

## 📚 Next Steps

- **Custom Training**: [TRAINING_FROM_SCRATCH.md](TRAINING_FROM_SCRATCH.md)
- **API Integration**: [API_REFERENCE.md](API_REFERENCE.md)
- **Advanced Features**: [USER_GUIDE.md](USER_GUIDE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Need Help?**
- GitHub Issues: https://github.com/yourusername/SwellSight_Colab/issues
- Documentation: https://swellsight.readthedocs.io
- Community: https://discord.gg/swellsight
