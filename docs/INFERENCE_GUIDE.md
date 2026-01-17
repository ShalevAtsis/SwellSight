# Wave Analysis Inference Guide

Complete guide for using the trained SwellSight model for beach cam analysis.

---

## 📋 Overview

This guide covers:
- Single image inference
- Batch processing
- Real-time video analysis
- Confidence interpretation
- Quality validation
- Performance optimization

---

## 🎯 Prerequisites

- Trained model checkpoint (see [TRAINING_FROM_SCRATCH.md](TRAINING_FROM_SCRATCH.md))
- SwellSight installed (`pip install -r requirements/base.txt`)
- Beach cam images or video

---

## 🖼️ Single Image Inference

### Basic Usage

```python
from src.swellsight.core.wave_analyzer import DINOv2WaveAnalyzer
from src.swellsight.core.depth_extractor import DepthAnythingV2Extractor
import cv2

# Initialize components
depth_extractor = DepthAnythingV2Extractor(
    model_size="large",
    precision="fp16",
    enable_optimization=True
)

wave_analyzer = DINOv2WaveAnalyzer(
    backbone_model="dinov2_vitb14",
    freeze_backbone=True,
    enable_optimization=True,
    confidence_calibration_method="isotonic"
)

# Load and preprocess image
image = cv2.imread('beach_cam.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract depth map
depth_map, depth_perf = depth_extractor.extract_depth(image_rgb)

# Analyze waves
wave_metrics, wave_perf, quality_results = wave_analyzer.analyze_waves(
    image_rgb, depth_map
)

# Display results
print(f"🌊 Wave Height: {wave_metrics.height_meters:.2f}m ({wave_metrics.height_feet:.1f}ft)")
print(f"🧭 Direction: {wave_metrics.direction}")
print(f"💥 Breaking Type: {wave_metrics.breaking_type}")
print(f"⚠️ Extreme Conditions: {wave_metrics.extreme_conditions}")
```


### Understanding Confidence Scores

```python
# Get comprehensive confidence metrics
confidence_scores = wave_analyzer.get_confidence_scores()

print(f"\n📊 Confidence Analysis:")
print(f"Overall: {confidence_scores.overall_confidence:.1%}")
print(f"Height: {confidence_scores.height_confidence:.1%}")
print(f"Direction: {confidence_scores.direction_confidence:.1%}")
print(f"Breaking: {confidence_scores.breaking_type_confidence:.1%}")

# Access detailed metrics
if confidence_scores.height_metrics:
    print(f"\nHeight Confidence Details:")
    print(f"  Calibrated: {confidence_scores.height_metrics.calibrated_confidence:.1%}")
    print(f"  Uncertainty: {confidence_scores.height_metrics.uncertainty:.3f}")
    print(f"  Reliability: {confidence_scores.height_metrics.reliability_score:.1%}")

# Interpret confidence levels
if confidence_scores.overall_confidence > 0.8:
    print("\n✓ High confidence - Reliable prediction")
elif confidence_scores.overall_confidence > 0.6:
    print("\n⚠️ Moderate confidence - Generally reliable")
else:
    print("\n❌ Low confidence - Verify visually")
```

### Quality Validation

```python
# Check quality validation results
if quality_results['input_rejected']:
    print("\n❌ Input Quality Issues:")
    print(f"  Image valid: {quality_results['input_validation']['image_valid']}")
    print(f"  Depth valid: {quality_results['input_validation']['depth_valid']}")
    print(f"  Reasons: {quality_results['input_validation'].get('rejection_reasons', [])}")
else:
    print("\n✓ Input quality validated")

# Check prediction quality
if quality_results['prediction_validation']:
    pred_val = quality_results['prediction_validation']
    if pred_val['is_valid']:
        print("✓ Prediction quality validated")
    else:
        print("⚠️ Anomalous prediction detected:")
        anomaly = pred_val['anomaly_metrics']
        print(f"  Anomaly score: {anomaly.anomaly_score:.3f}")
        print(f"  Reasons: {anomaly.anomaly_reasons}")

# Check performance
if quality_results['performance_monitoring']:
    perf_mon = quality_results['performance_monitoring']
    if perf_mon.is_degraded:
        print("\n⚠️ Performance degradation detected:")
        print(f"  Reasons: {perf_mon.degradation_reasons}")
```

### Performance Metrics

```python
# Check inference performance
if wave_perf:
    print(f"\n⚡ Performance Metrics:")
    print(f"Total time: {wave_perf.total_time_ms:.1f}ms")
    print(f"  Preprocessing: {wave_perf.preprocessing_time_ms:.1f}ms")
    print(f"  Model forward: {wave_perf.forward_time_ms:.1f}ms")
    print(f"  Postprocessing: {wave_perf.postprocessing_time_ms:.1f}ms")
    print(f"Memory usage: {wave_perf.memory_usage_mb:.1f}MB")
    
    # Check if real-time capable
    if wave_analyzer.is_real_time_capable():
        print("✓ Real-time processing capable")
    else:
        print("⚠️ Not meeting real-time requirements")
```

---

## 📁 Batch Processing

### Process Multiple Images

```python
from pathlib import Path
import json
from tqdm import tqdm

def process_batch(image_dir, output_file='results.json'):
    """Process all images in a directory."""
    
    # Initialize components once
    depth_extractor = DepthAnythingV2Extractor(model_size="large")
    wave_analyzer = DINOv2WaveAnalyzer(enable_optimization=True)
    
    # Find all images
    image_paths = list(Path(image_dir).glob('*.jpg'))
    image_paths += list(Path(image_dir).glob('*.png'))
    
    results = []
    
    print(f"Processing {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract depth
            depth_map, _ = depth_extractor.extract_depth(image_rgb)
            
            # Analyze waves
            wave_metrics, perf, quality = wave_analyzer.analyze_waves(
                image_rgb, depth_map
            )
            
            # Store results
            result = {
                'filename': img_path.name,
                'timestamp': img_path.stat().st_mtime,
                'wave_metrics': {
                    'height_meters': float(wave_metrics.height_meters),
                    'height_feet': float(wave_metrics.height_feet),
                    'direction': wave_metrics.direction,
                    'breaking_type': wave_metrics.breaking_type,
                    'extreme_conditions': wave_metrics.extreme_conditions
                },
                'confidence': {
                    'height': float(wave_metrics.height_confidence),
                    'direction': float(wave_metrics.direction_confidence),
                    'breaking': float(wave_metrics.breaking_confidence)
                },
                'quality': {
                    'input_valid': not quality['input_rejected'],
                    'prediction_valid': quality['prediction_validation']['is_valid'] 
                        if quality['prediction_validation'] else None
                },
                'performance': {
                    'inference_time_ms': float(perf.total_time_ms) if perf else None
                }
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Processed {len(results)} images")
    print(f"✓ Results saved to {output_file}")
    
    # Print summary statistics
    heights = [r['wave_metrics']['height_meters'] for r in results]
    print(f"\nSummary Statistics:")
    print(f"  Average height: {sum(heights)/len(heights):.2f}m")
    print(f"  Min height: {min(heights):.2f}m")
    print(f"  Max height: {max(heights):.2f}m")
    
    return results

# Usage
results = process_batch('beach_cams/', 'wave_analysis_results.json')
```

### Parallel Batch Processing

For faster processing with multiple GPUs or CPU cores:

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def process_single_image(img_path, depth_extractor, wave_analyzer):
    """Process a single image (for parallel execution)."""
    try:
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth_map, _ = depth_extractor.extract_depth(image_rgb)
        wave_metrics, _, _ = wave_analyzer.analyze_waves(image_rgb, depth_map)
        
        return {
            'filename': img_path.name,
            'height_meters': float(wave_metrics.height_meters),
            'direction': wave_metrics.direction,
            'breaking_type': wave_metrics.breaking_type
        }
    except Exception as e:
        return {'filename': img_path.name, 'error': str(e)}

def parallel_batch_process(image_dir, num_workers=4):
    """Process images in parallel."""
    
    # Initialize components
    depth_extractor = DepthAnythingV2Extractor(model_size="large")
    wave_analyzer = DINOv2WaveAnalyzer(enable_optimization=True)
    
    # Find images
    image_paths = list(Path(image_dir).glob('*.jpg'))
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            lambda p: process_single_image(p, depth_extractor, wave_analyzer),
            image_paths
        ))
    
    return results

# Usage
results = parallel_batch_process('beach_cams/', num_workers=4)
```

---

## 🎥 Real-Time Video Analysis

### Process Video File

```python
def analyze_video(video_path, output_path=None, display=True):
    """Analyze waves in video file."""
    
    # Initialize components
    depth_extractor = DepthAnythingV2Extractor(model_size="large")
    wave_analyzer = DINOv2WaveAnalyzer(enable_optimization=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    results = []
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every Nth frame for efficiency
        if frame_count % 5 != 0:  # Process every 5th frame
            continue
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract depth and analyze
        depth_map, _ = depth_extractor.extract_depth(frame_rgb)
        wave_metrics, perf, _ = wave_analyzer.analyze_waves(frame_rgb, depth_map)
        
        # Store results
        results.append({
            'frame': frame_count,
            'timestamp': frame_count / fps,
            'height_meters': float(wave_metrics.height_meters),
            'direction': wave_metrics.direction,
            'breaking_type': wave_metrics.breaking_type,
            'inference_time_ms': float(perf.total_time_ms) if perf else None
        })
        
        # Overlay results on frame
        cv2.putText(frame, f"Height: {wave_metrics.height_meters:.1f}m", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Direction: {wave_metrics.direction}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Breaking: {wave_metrics.breaking_type}", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {wave_metrics.height_confidence:.0%}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write output frame
        if output_path:
            out.write(frame)
        
        # Display frame
        if display:
            cv2.imshow('Wave Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"\n✓ Processed {len(results)} frames")
    
    # Print summary
    if results:
        avg_height = sum(r['height_meters'] for r in results) / len(results)
        avg_time = sum(r['inference_time_ms'] for r in results if r['inference_time_ms']) / len(results)
        print(f"Average wave height: {avg_height:.2f}m")
        print(f"Average inference time: {avg_time:.1f}ms")
    
    return results

# Usage
results = analyze_video(
    'beach_cam_video.mp4',
    output_path='analyzed_video.mp4',
    display=True
)
```

### Live Webcam Analysis

```python
def analyze_webcam(camera_id=0):
    """Analyze waves from live webcam feed."""
    
    # Initialize components
    depth_extractor = DepthAnythingV2Extractor(model_size="large")
    wave_analyzer = DINOv2WaveAnalyzer(enable_optimization=True)
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    
    print("Starting live wave analysis...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract depth and analyze
        depth_map, _ = depth_extractor.extract_depth(frame_rgb)
        wave_metrics, perf, quality = wave_analyzer.analyze_waves(frame_rgb, depth_map)
        
        # Overlay results
        cv2.putText(frame, f"Height: {wave_metrics.height_meters:.1f}m", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Direction: {wave_metrics.direction}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Breaking: {wave_metrics.breaking_type}", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show FPS
        if perf:
            fps = 1000 / perf.total_time_ms
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Live Wave Analysis', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Usage
analyze_webcam(camera_id=0)
```

---

## 🎯 Advanced Inference Techniques

### Confidence Calibration

Improve confidence score accuracy:

```python
# Collect calibration data from validation set
calibration_data = []

for image, depth, ground_truth in validation_dataset:
    wave_metrics, _, _ = wave_analyzer.analyze_waves(image, depth)
    
    # Add calibration data
    wave_analyzer.add_confidence_calibration_data(wave_metrics, ground_truth)
    
    calibration_data.append({
        'predicted': wave_metrics,
        'ground_truth': ground_truth
    })

# Fit calibrators (requires minimum 50 samples)
calibration_status = wave_analyzer.fit_confidence_calibrators(min_samples=50)

print("Calibration Status:")
for task, fitted in calibration_status.items():
    print(f"  {task}: {'✓ Fitted' if fitted else '✗ Not enough data'}")

# Analyze calibration quality
for task in ["height", "direction", "breaking"]:
    cal_results = wave_analyzer.analyze_confidence_calibration(task)
    print(f"\n{task.capitalize()} Calibration:")
    print(f"  ECE: {cal_results.expected_calibration_error:.3f}")
    print(f"  MCE: {cal_results.max_calibration_error:.3f}")
    print(f"  Brier Score: {cal_results.brier_score:.3f}")

# Save calibration for production use
wave_analyzer.save_confidence_calibration_data('calibration_data.pkl')
```

### Ensemble Predictions

Combine multiple models for better accuracy:

```python
def ensemble_predict(image, depth, models):
    """Ensemble prediction from multiple models."""
    
    predictions = []
    
    for model in models:
        wave_metrics, _, _ = model.analyze_waves(image, depth)
        predictions.append(wave_metrics)
    
    # Average height predictions
    avg_height = sum(p.height_meters for p in predictions) / len(predictions)
    
    # Majority vote for direction
    from collections import Counter
    direction_votes = Counter(p.direction for p in predictions)
    direction = direction_votes.most_common(1)[0][0]
    
    # Majority vote for breaking type
    breaking_votes = Counter(p.breaking_type for p in predictions)
    breaking_type = breaking_votes.most_common(1)[0][0]
    
    # Average confidence
    avg_confidence = sum(p.height_confidence for p in predictions) / len(predictions)
    
    return {
        'height_meters': avg_height,
        'direction': direction,
        'breaking_type': breaking_type,
        'confidence': avg_confidence,
        'num_models': len(models)
    }

# Usage
models = [
    DINOv2WaveAnalyzer(backbone_model="dinov2_vitb14"),
    DINOv2WaveAnalyzer(backbone_model="dinov2_vitl14"),
]

result = ensemble_predict(image, depth, models)
print(f"Ensemble prediction: {result}")
```

---

## 📊 Result Interpretation

### Wave Height Guidelines

| Height (m) | Height (ft) | Surfability | Skill Level |
|------------|-------------|-------------|-------------|
| 0.3 - 0.6 | 1 - 2 | Flat | Beginner practice |
| 0.6 - 1.2 | 2 - 4 | Small | Beginner |
| 1.2 - 1.8 | 4 - 6 | Medium | Intermediate |
| 1.8 - 2.4 | 6 - 8 | Large | Advanced |
| 2.4+ | 8+ | Very Large | Expert |

### Direction Interpretation

- **LEFT**: Wave breaks from right to left (surfer's perspective facing shore)
- **RIGHT**: Wave breaks from left to right
- **STRAIGHT**: Wave breaks straight toward shore (closeout - not ideal)

### Breaking Type Characteristics

- **SPILLING**: Gentle, foamy break - Best for beginners
- **PLUNGING**: Hollow, barreling wave - Best for advanced surfers
- **SURGING**: Breaks directly on shore - Not surfable
- **NO_BREAKING**: No clear breaking pattern detected

---

## 🔍 Quality Assurance

### Input Quality Checks

```python
from src.swellsight.utils.quality_validation import ComprehensiveQualityValidator

validator = ComprehensiveQualityValidator()

# Validate input before processing
is_valid, metrics = validator.validate_input_quality(image, depth_map)

if not is_valid:
    print("❌ Input quality issues:")
    if not metrics['image_valid']:
        print(f"  Image issues: {metrics.get('image_issues', [])}")
    if not metrics['depth_valid']:
        print(f"  Depth issues: {metrics.get('depth_issues', [])}")
else:
    print("✓ Input quality acceptable")
    # Proceed with inference
```

### Prediction Anomaly Detection

```python
# Fit anomaly detector on historical predictions
historical_predictions = [
    {'height_meters': 1.5, 'direction_confidence': 0.85, ...},
    {'height_meters': 2.1, 'direction_confidence': 0.92, ...},
    # ... more predictions
]

wave_analyzer.fit_anomaly_detector(historical_predictions)

# Check for anomalies in new predictions
wave_metrics, _, quality = wave_analyzer.analyze_waves(image, depth)

if quality['prediction_validation']:
    if not quality['prediction_validation']['is_valid']:
        print("⚠️ Anomalous prediction detected!")
        anomaly = quality['prediction_validation']['anomaly_metrics']
        print(f"Anomaly score: {anomaly.anomaly_score:.3f}")
        print(f"Reasons: {anomaly.anomaly_reasons}")
```

---

## 📈 Performance Monitoring

### Track Inference Performance

```python
# Monitor performance over time
performance_history = []

for image, depth in test_dataset:
    wave_metrics, perf, _ = wave_analyzer.analyze_waves(image, depth)
    
    if perf:
        performance_history.append({
            'total_time_ms': perf.total_time_ms,
            'memory_mb': perf.memory_usage_mb
        })

# Analyze performance
import numpy as np
times = [p['total_time_ms'] for p in performance_history]
memory = [p['memory_mb'] for p in performance_history]

print(f"Performance Statistics:")
print(f"  Avg inference time: {np.mean(times):.1f}ms")
print(f"  Min inference time: {np.min(times):.1f}ms")
print(f"  Max inference time: {np.max(times):.1f}ms")
print(f"  Std inference time: {np.std(times):.1f}ms")
print(f"  Avg memory usage: {np.mean(memory):.1f}MB")
```

### System Health Monitoring

```python
# Get comprehensive system health report
health_report = wave_analyzer.get_system_health_report()

print("System Health Report:")
print(f"  Status: {health_report['status']}")
print(f"  Uptime: {health_report['uptime_seconds']:.0f}s")
print(f"  Total predictions: {health_report['total_predictions']}")
print(f"  Error rate: {health_report['error_rate']:.1%}")
print(f"  Avg confidence: {health_report['avg_confidence']:.1%}")

if health_report['alerts']:
    print("\n⚠️ Alerts:")
    for alert in health_report['alerts']:
        print(f"  - {alert}")
```

---

## 💡 Best Practices

1. **Always validate input quality** before inference
2. **Monitor confidence scores** - low confidence may indicate poor input
3. **Use calibrated confidence** for production systems
4. **Enable optimization** for real-time performance
5. **Batch process** when possible for efficiency
6. **Monitor system health** in production deployments
7. **Save results** with timestamps for analysis
8. **Handle errors gracefully** - don't crash on bad input

---

## 📚 Next Steps

- **Training Guide**: [TRAINING_FROM_SCRATCH.md](TRAINING_FROM_SCRATCH.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
