# Design Document

## Overview

The SwellSight Pipeline Improvements design extends the pipeline from 8 to 13 notebooks, adding comprehensive wave analysis capabilities while maintaining simplicity and notebook-based architecture. The design encompasses three phases:

**Phase 1: Foundation Pipeline (Notebooks 01-08)** - Enhanced data flow integration, memory optimization, quality validation, error handling, progress tracking, model training efficiency, evaluation metrics, and configuration management. Updates include Depth-Anything-V2 for depth extraction and FLUX.1-dev for synthetic generation.

**Phase 2: Wave Analysis Extension (Notebooks 09-12)** - Implements the complete Wave Analyzer system with DINOv2 backbone integration, multi-task model architecture, sim-to-real training strategy, and real-time wave metrics inference.

**Phase 3: Comprehensive Evaluation (Notebook 13)** - Provides comprehensive evaluation framework for wave height prediction, direction classification, and breaking type classification on real beach cam data.

## Architecture

### High-Level Architecture

The improved pipeline extends from 8 to 13 notebooks, organized in three phases: Foundation Pipeline (01-08), Wave Analysis Extension (09-12), and Comprehensive Evaluation (13). The architecture maintains shared utility functions, standardized data formats, and improved error handling:

```
Phase 1: Foundation Pipeline
01_Setup_and_Installation.ipynb
    ↓ (config.json, environment_info.json)
02_Data_Import_and_Preprocessing.ipynb  
    ↓ (processed_images.json, quality_report.json)
03_Depth_Anything_V2_Extraction.ipynb
    ↓ (depth_maps.json, depth_quality.json)
04_Data_Augmentation_System.ipynb
    ↓ (augmentation_params.json)
05_FLUX_ControlNet_Synthetic_Generation.ipynb
    ↓ (synthetic_images.json, synthetic_labels.json, generation_report.json)
06_Model_Training_Pipeline.ipynb
    ↓ (trained_model.pth, training_metrics.json)
07_Exploratory_Data_Analysis.ipynb
    ↓ (analysis_report.json, visualizations/)
08_Model_Evaluation_and_Validation.ipynb
    ↓ (evaluation_results.json, baseline_report.html)

Phase 2: Wave Analysis Extension
09_DINOv2_Backbone_Integration.ipynb
    ↓ (dinov2_backbone.pth, feature_validation.json)
10_Multi_Task_Model_Architecture.ipynb
    ↓ (multi_task_model.pth, architecture_config.json)
11_Wave_Analyzer_Training.ipynb
    ↓ (wave_analyzer.pth, training_history.json, best_checkpoint.pth)
12_Wave_Metrics_Inference.ipynb
    ↓ (wave_predictions.json, inference_visualizations/)

Phase 3: Comprehensive Evaluation
13_Wave_Analysis_Evaluation.ipynb
    ↓ (evaluation_results.json, confusion_matrices/, final_report.html)
```

### Shared Components

Each notebook will include common utility functions for:
- **Data Validation**: Standardized validation functions for file existence, format checking, and quality assessment
- **Memory Management**: Dynamic batch sizing, memory monitoring, and cleanup utilities
- **Error Handling**: Retry logic, fallback mechanisms, and user-friendly error messages
- **Progress Tracking**: Consistent progress bars and status reporting across notebooks
- **Configuration Management**: Centralized configuration loading and validation

### Wave Analysis Architecture

The Wave Analyzer implements a hybrid pipeline where generative AI trains analytical AI:

**Stage A: Depth Extraction (The "Eye")**
- Depth-Anything-V2-Large converts 2D beach cam images into high-sensitivity depth maps
- Captures fine-grained wave texture and geometry
- Preserves sharp wave edges and far-field depth sensitivity

**Stage B: Synthetic Data Factory (The "Simulator")**
- FLUX.1-dev + ControlNet-Depth generates photorealistic synthetic images
- Automatic label generation for wave height, direction, and breaking type
- Solves manual labeling challenges through sim-to-real training

**Stage C: Wave Analyzer (The "Brain")**
- DINOv2 backbone provides geometric intelligence for wave understanding
- Multi-task model with three specialized heads predicts all wave metrics
- Trained using sim-to-real strategy: synthetic pre-training + real data fine-tuning

## Components and Interfaces

### 1. Data Flow Manager

**Purpose**: Ensures smooth data transfer between notebooks with validation and error handling.

**Key Functions**:
- `save_stage_results(data, stage_name, metadata)`: Standardized result saving
- `load_previous_results(stage_name, required_files)`: Validated data loading
- `validate_data_format(data, expected_schema)`: Format validation
- `check_dependencies(required_stages)`: Dependency verification

**Interface**:
```python
# Save results with metadata
save_stage_results(
    data={"images": image_paths, "quality_scores": scores},
    stage_name="depth_extraction", 
    metadata={"model_used": "dpt-large", "processing_time": 1200}
)

# Load and validate previous results
results = load_previous_results(
    stage_name="data_preprocessing",
    required_files=["processed_images.json", "quality_report.json"]
)
```

### 2. Memory Optimizer

**Purpose**: Manages memory usage dynamically based on available resources.

**Key Functions**:
- `get_optimal_batch_size(available_memory, item_size)`: Calculate optimal batch size
- `monitor_memory_usage()`: Track current memory consumption
- `cleanup_variables(variable_list)`: Explicit memory cleanup
- `suggest_memory_optimizations()`: Provide optimization recommendations

**Interface**:
```python
# Dynamic batch sizing
batch_size = get_optimal_batch_size(
    available_memory=torch.cuda.get_device_properties(0).total_memory,
    item_size=estimate_image_memory_usage(image_resolution)
)

# Memory monitoring and cleanup
with memory_monitor() as monitor:
    process_batch(images)
    if monitor.usage > 0.8:
        cleanup_variables([large_tensor1, large_tensor2])
```

### 3. Quality Validator

**Purpose**: Implements consistent quality checks across all pipeline stages.

**Key Functions**:
- `validate_image_quality(image_path)`: Basic image quality metrics
- `validate_depth_map_quality(depth_map)`: Depth map quality assessment
- `compare_data_distributions(real_data, synthetic_data)`: Statistical comparison
- `generate_quality_report(quality_metrics)`: Standardized quality reporting

**Interface**:
```python
# Image quality validation
quality_metrics = validate_image_quality(image_path)
if quality_metrics['score'] < quality_threshold:
    log_warning(f"Low quality image: {image_path}")
    
# Generate quality report
report = generate_quality_report({
    'total_images': len(images),
    'quality_scores': quality_scores,
    'failed_validations': failed_count
})
```

### 4. Error Handler

**Purpose**: Provides consistent error handling and recovery mechanisms.

**Key Functions**:
- `retry_with_backoff(func, max_retries, backoff_factor)`: Exponential backoff retry
- `handle_gpu_memory_error(operation, fallback_func)`: GPU error handling
- `save_partial_results(results, stage_name)`: Partial result preservation
- `provide_recovery_instructions(error_type)`: User guidance for recovery

**Interface**:
```python
# Retry with exponential backoff
result = retry_with_backoff(
    func=lambda: download_model(model_url),
    max_retries=3,
    backoff_factor=2.0
)

# GPU error handling with CPU fallback
try:
    result = process_on_gpu(data)
except torch.cuda.OutOfMemoryError:
    result = handle_gpu_memory_error(
        operation="depth_extraction",
        fallback_func=lambda: process_on_cpu(data)
    )
```

### 5. Progress Tracker

**Purpose**: Provides consistent progress tracking and user feedback.

**Key Functions**:
- `create_progress_bar(total_items, description)`: Standardized progress bars
- `update_progress(current, total, additional_info)`: Progress updates
- `display_stage_summary(metrics)`: Stage completion summaries
- `show_performance_tips(current_performance)`: Performance optimization suggestions

**Interface**:
```python
# Progress tracking with additional info
with create_progress_bar(len(images), "Processing images") as pbar:
    for i, image in enumerate(images):
        result = process_image(image)
        pbar.update(1, additional_info=f"Memory: {get_memory_usage():.1f}GB")
```

### 6. DINOv2 Backbone Adapter

**Purpose**: Integrates DINOv2 self-supervised vision transformer for geometric feature extraction.

**Key Functions**:
- `load_dinov2_backbone(variant="vit_large")`: Load pre-trained DINOv2 model
- `adapt_input_channels(model, num_channels=4)`: Adapt for RGB + Depth input
- `extract_features(images, depth_maps)`: Extract 1024-dim feature vectors
- `validate_feature_quality(features)`: Verify feature quality and similarity

**Interface**:
```python
# Load and adapt DINOv2 backbone
backbone = load_dinov2_backbone(variant="vit_large")
backbone = adapt_input_channels(backbone, num_channels=4)  # RGB + Depth

# Extract features with frozen backbone
with torch.no_grad():
    features = extract_features(
        images=beach_cam_images,
        depth_maps=depth_maps
    )  # Output: (batch_size, 1024)
```

**Design Rationale**: DINOv2 provides superior geometric understanding through self-supervised learning on diverse visual data. The frozen backbone preserves pre-trained knowledge while the 4-channel adaptation allows depth information integration.

### 7. Multi-Task Model

**Purpose**: Implements multi-task neural network with three specialized prediction heads.

**Key Functions**:
- `create_multi_task_model(backbone, num_classes_per_task)`: Build model architecture
- `forward_pass(features)`: Generate predictions for all three tasks
- `compute_weighted_loss(predictions, targets, task_weights)`: Balanced multi-task loss
- `get_confidence_scores(predictions)`: Extract prediction confidence

**Interface**:
```python
# Create multi-task model
model = create_multi_task_model(
    backbone=dinov2_backbone,
    num_classes_per_task={
        "height": 1,  # Regression
        "direction": 3,  # Left, Right, Straight
        "breaking_type": 3  # Spilling, Plunging, Surging
    }
)

# Forward pass with all predictions
predictions = model(features)
# predictions = {
#     "height": tensor([1.5]),  # meters
#     "direction": tensor([0.1, 0.8, 0.1]),  # probabilities
#     "breaking_type": tensor([0.7, 0.2, 0.1]),  # probabilities
#     "confidence": {"height": 0.85, "direction": 0.92, "breaking_type": 0.78}
# }
```

**Design Rationale**: Multi-task learning enables efficient prediction of all wave metrics from shared features. Task-specific projection layers prevent negative transfer, while weighted loss functions balance task importance during training.

### 8. Wave Analyzer Training Manager

**Purpose**: Implements sim-to-real training strategy with synthetic pre-training and real data fine-tuning.

**Key Functions**:
- `pretrain_on_synthetic(model, synthetic_data, epochs=50)`: Synthetic data pre-training
- `finetune_on_real(model, real_data, epochs=10)`: Real data fine-tuning
- `apply_learning_rate_schedule(optimizer, schedule_type="cosine_annealing")`: LR scheduling
- `track_per_task_losses(losses)`: Monitor individual task performance
- `save_best_checkpoint(model, metrics, checkpoint_dir)`: Checkpoint management

**Interface**:
```python
# Sim-to-real training pipeline
model = create_multi_task_model(backbone, num_classes_per_task)

# Phase 1: Pre-train on synthetic data
pretrain_metrics = pretrain_on_synthetic(
    model=model,
    synthetic_data=synthetic_dataset,
    epochs=50,
    lr_schedule="cosine_annealing"
)

# Phase 2: Fine-tune on real data
finetune_metrics = finetune_on_real(
    model=model,
    real_data=real_beach_cam_dataset,
    epochs=10,
    lr_schedule="cosine_annealing_warm_restart"
)

# Save best model
save_best_checkpoint(model, finetune_metrics, "./checkpoints")
```

**Design Rationale**: Sim-to-real training solves the manual labeling challenge by leveraging synthetic data with perfect labels. Pre-training establishes strong feature representations, while fine-tuning adapts to real-world beach cam characteristics.

### 9. Wave Metrics Inference Engine

**Purpose**: Provides end-to-end inference pipeline for real-time wave analysis.

**Key Functions**:
- `load_trained_model(checkpoint_path)`: Load best model checkpoint
- `process_beach_cam_image(image_path)`: End-to-end processing pipeline
- `extract_wave_metrics(predictions)`: Format predictions as wave metrics
- `visualize_predictions(image, predictions)`: Create annotated visualizations
- `batch_inference(image_paths)`: Process multiple images efficiently

**Interface**:
```python
# Load trained Wave Analyzer
wave_analyzer = load_trained_model("./checkpoints/best_model.pth")

# Single image inference
metrics = process_beach_cam_image(
    image_path="beach_cam_001.jpg",
    model=wave_analyzer
)
# metrics = {
#     "wave_height": {"meters": 1.5, "feet": 4.9, "confidence": 0.85},
#     "direction": {"class": "Right", "confidence": 0.92},
#     "breaking_type": {"class": "Spilling", "confidence": 0.78}
# }

# Visualize results
visualize_predictions(
    image="beach_cam_001.jpg",
    predictions=metrics,
    output_path="annotated_001.jpg"
)
```

**Design Rationale**: The inference engine provides a simple interface for surfers to get actionable wave metrics. Confidence scores enable users to assess prediction reliability, while visualizations make results intuitive.

### 10. Wave Analysis Evaluator

**Purpose**: Comprehensive evaluation framework for wave analysis performance.

**Key Functions**:
- `evaluate_wave_height(predictions, ground_truth)`: Compute MAE, RMSE, R²
- `evaluate_direction(predictions, ground_truth)`: Compute accuracy, precision, recall, F1
- `evaluate_breaking_type(predictions, ground_truth)`: Compute confusion matrix and per-class metrics
- `quantify_sim_to_real_gap(synthetic_performance, real_performance)`: Transfer gap analysis
- `generate_evaluation_report(all_metrics)`: Create comprehensive report

**Interface**:
```python
# Evaluate on real beach cam test set
height_metrics = evaluate_wave_height(
    predictions=predicted_heights,
    ground_truth=true_heights
)
# height_metrics = {"MAE": 0.18, "RMSE": 0.24, "R2": 0.89}

direction_metrics = evaluate_direction(
    predictions=predicted_directions,
    ground_truth=true_directions
)
# direction_metrics = {
#     "accuracy": 0.91,
#     "per_class": {"Left": {"precision": 0.89, "recall": 0.92, "f1": 0.90}, ...}
# }

# Generate comprehensive report
report = generate_evaluation_report({
    "height": height_metrics,
    "direction": direction_metrics,
    "breaking_type": breaking_type_metrics,
    "sim_to_real_gap": gap_metrics
})
```

**Design Rationale**: Comprehensive evaluation ensures the Wave Analyzer meets accuracy targets. Per-task metrics enable targeted improvements, while sim-to-real gap analysis validates the training strategy.

## Data Models

### Configuration Schema

```json
{
  "pipeline": {
    "name": "swellsight_pipeline",
    "version": "1.0",
    "created": "2024-01-01T00:00:00Z"
  },
  "processing": {
    "batch_size": "auto",
    "max_images": 1000,
    "quality_threshold": 0.7,
    "memory_limit_gb": "auto"
  },
  "models": {
    "depth_model": "depth-anything/Depth-Anything-V2-Large",
    "base_model": "black-forest-labs/FLUX.1-dev",
    "controlnet_model": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
    "mixed_precision": true
  },
  "paths": {
    "data_dir": "./data",
    "output_dir": "./outputs",
    "checkpoint_dir": "./checkpoints"
  }
}
```

### Stage Result Schema

```json
{
  "stage_name": "depth_anything_v2_extraction",
  "timestamp": "2024-01-01T12:00:00Z",
  "status": "completed",
  "processing_time_seconds": 1200,
  "input_count": 500,
  "output_count": 495,
  "success_rate": 0.99,
  "quality_metrics": {
    "mean_quality_score": 0.85,
    "min_quality_score": 0.45,
    "max_quality_score": 0.98
  },
  "errors": [
    {
      "type": "corrupted_image",
      "count": 3,
      "examples": ["image_001.jpg", "image_045.jpg"]
    }
  ],
  "outputs": {
    "depth_maps": "depth_maps/",
    "metadata": "depth_anything_v2_metadata.json"
  }
}
```

### Wave Metrics Output Schema

```json
{
  "image_id": "beach_cam_001.jpg",
  "timestamp": "2024-01-01T14:30:00Z",
  "wave_metrics": {
    "wave_height": {
      "meters": 1.5,
      "feet": 4.9,
      "confidence": 0.85,
      "quality_flag": "high"
    },
    "direction": {
      "class": "Right",
      "probabilities": {
        "Left": 0.08,
        "Right": 0.84,
        "Straight": 0.08
      },
      "confidence": 0.92,
      "quality_flag": "high"
    },
    "breaking_type": {
      "class": "Spilling",
      "probabilities": {
        "Spilling": 0.72,
        "Plunging": 0.18,
        "Surging": 0.10
      },
      "confidence": 0.78,
      "quality_flag": "medium"
    }
  },
  "processing_info": {
    "depth_extraction_time_ms": 450,
    "feature_extraction_time_ms": 320,
    "inference_time_ms": 180,
    "total_time_ms": 950,
    "model_version": "wave_analyzer_v1.0"
  }
}
```

### Synthetic Label Schema

```json
{
  "image_id": "synthetic_001.jpg",
  "generation_params": {
    "prompt": "Ocean waves breaking on beach, spilling waves, 1.5m height, breaking right",
    "depth_conditioning_strength": 0.8,
    "guidance_scale": 7.5,
    "num_inference_steps": 50
  },
  "ground_truth_labels": {
    "wave_height_meters": 1.5,
    "direction": "Right",
    "breaking_type": "Spilling"
  },
  "quality_metrics": {
    "depth_map_quality": 0.92,
    "image_quality": 0.88,
    "realism_score": 0.85
  }
}
```

### Multi-Task Model Architecture Schema

```json
{
  "model_name": "wave_analyzer_multi_task",
  "version": "1.0",
  "architecture": {
    "backbone": {
      "type": "dinov2_vit_large",
      "input_channels": 4,
      "output_features": 1024,
      "frozen": true
    },
    "heads": {
      "wave_height": {
        "type": "regression",
        "layers": [1024, 512, 256, 1],
        "activation": "relu",
        "output_activation": "relu"
      },
      "direction": {
        "type": "classification",
        "num_classes": 3,
        "layers": [1024, 512, 256, 3],
        "activation": "relu",
        "output_activation": "softmax"
      },
      "breaking_type": {
        "type": "classification",
        "num_classes": 3,
        "layers": [1024, 512, 256, 3],
        "activation": "relu",
        "output_activation": "softmax"
      }
    }
  },
  "training_config": {
    "loss_weights": {
      "wave_height": 1.0,
      "direction": 1.5,
      "breaking_type": 1.2
    },
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "scheduler": "cosine_annealing_warm_restart"
  }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Data Format Consistency
*For any* notebook stage completion, saving results and then loading them should preserve data integrity and format compliance
**Validates: Requirements 1.1**

### Property 2: Input Validation Completeness  
*For any* data loading operation, providing invalid inputs (missing files, corrupted data, wrong formats) should trigger appropriate validation errors
**Validates: Requirements 1.2**

### Property 3: Configuration Loading Reliability
*For any* notebook startup, configuration files should be properly loaded and parsed, with invalid configurations being rejected with clear error messages
**Validates: Requirements 1.3**

### Property 4: Error Message Clarity
*For any* data loading error condition, the system should generate clear, actionable error messages with suggested fixes
**Validates: Requirements 1.4**

### Property 5: Dependency Detection Accuracy
*For any* notebook execution order, missing dependencies should be detected and appropriate user guidance should be provided
**Validates: Requirements 1.5**

### Property 6: Dynamic Batch Sizing Adaptation
*For any* available memory configuration, batch sizes should be dynamically adjusted to optimize memory usage without exceeding limits
**Validates: Requirements 2.1**

### Property 7: Memory-Based Batch Reduction
*For any* low GPU memory condition, batch sizes should be automatically reduced with appropriate user warnings
**Validates: Requirements 2.2**

### Property 8: Model Memory Cleanup
*For any* model loading sequence, previous models should be cleared from memory to prevent accumulation
**Validates: Requirements 2.3**

### Property 9: Processing Memory Cleanup
*For any* processing completion, large variables should be explicitly cleared and garbage collection should be called
**Validates: Requirements 2.4**

### Property 10: Memory Usage Warnings
*For any* high memory usage condition, appropriate warnings and optimization suggestions should be displayed
**Validates: Requirements 2.5**

### Property 11: Image Quality Validation
*For any* image processing operation, basic quality metrics (resolution, format, corruption) should be checked and validated
**Validates: Requirements 3.1**

### Property 12: Depth Map Quality Assessment
*For any* depth map generation, quality should be validated using statistical measures with appropriate thresholds
**Validates: Requirements 3.2**

### Property 13: Synthetic Data Distribution Comparison
*For any* synthetic image creation, basic statistical comparisons between real and synthetic data should be performed
**Validates: Requirements 3.3**

### Property 14: Quality-Based Filtering
*For any* quality issue detection, appropriate warnings should be logged and poor-quality samples should be optionally excluded
**Validates: Requirements 3.4**

### Property 15: Quality Summary Reporting
*For any* processing completion, a summary of quality statistics and issues should be displayed
**Validates: Requirements 3.5**

### Property 16: File Operation Retry Logic
*For any* file operation failure, retry attempts with exponential backoff should be implemented with clear error messages
**Validates: Requirements 4.1**

### Property 17: Individual Image Fault Tolerance
*For any* batch processing with failed images, the system should skip failed items, log failures, and continue with remaining images
**Validates: Requirements 4.2**

### Property 18: GPU Fallback Mechanism
*For any* GPU operation failure, the system should fall back to CPU processing with appropriate warnings
**Validates: Requirements 4.3**

### Property 19: Network Download Resilience
*For any* network download failure, retry logic and fallback options should be provided
**Validates: Requirements 4.4**

### Property 20: Critical Error Recovery
*For any* critical error occurrence, partial results should be saved and recovery instructions should be provided
**Validates: Requirements 4.5**

### Property 21: Progress Bar Display
*For any* large batch processing, progress bars with time estimates should be displayed and updated correctly
**Validates: Requirements 5.1**

### Property 22: Stage Summary Generation
*For any* stage completion, summary statistics and processing metrics should be displayed
**Validates: Requirements 5.2**

### Property 23: Error Progress Tracking
*For any* error occurrence during processing, progress indicators should be updated to reflect error counts
**Validates: Requirements 5.3**

### Property 24: Memory Usage Progress Display
*For any* high memory usage condition, current memory usage should be displayed in progress updates
**Validates: Requirements 5.4**

### Property 25: Performance Optimization Guidance
*For any* slow processing condition, performance tips and optimization suggestions should be provided
**Validates: Requirements 5.5**

### Property 26: Mixed Precision Training Adaptation
*For any* training initialization, mixed precision training should be implemented when supported by hardware
**Validates: Requirements 6.1**

### Property 27: Efficient Data Loading
*For any* training data loading, efficient data loading with appropriate batch sizes should be used
**Validates: Requirements 6.2**

### Property 28: Checkpoint Management
*For any* training execution, checkpoints should be saved at configurable intervals with simple management
**Validates: Requirements 6.3**

### Property 29: Training Plateau Guidance
*For any* training metrics plateau condition, suggestions for learning rate adjustments should be provided
**Validates: Requirements 6.4**

### Property 30: Model Persistence with Metadata
*For any* training completion, the final model should be saved with comprehensive metadata
**Validates: Requirements 6.5**

### Property 31: Comprehensive Evaluation Metrics
*For any* model evaluation, both standard metrics and domain-specific wave analysis metrics should be computed
**Validates: Requirements 7.1**

### Property 32: Data Comparison Methodology
*For any* synthetic vs real data comparison, simple statistical comparisons and visualizations should be used
**Validates: Requirements 7.2**

### Property 33: Prediction Confidence Scoring
*For any* prediction generation, confidence scores should be included where applicable
**Validates: Requirements 7.3**

### Property 34: Evaluation Visualization and Reporting
*For any* evaluation completion, clear visualizations and summary reports should be created
**Validates: Requirements 7.4**

### Property 35: Performance Improvement Suggestions
*For any* poor performance detection, potential improvements should be suggested based on results
**Validates: Requirements 7.5**

### Property 36: Configuration Loading with Defaults
*For any* notebook startup, configuration should be loaded from JSON files with sensible defaults applied
**Validates: Requirements 8.1**

### Property 37: Parameter Validation and Warnings
*For any* parameter change, ranges should be validated and warnings should be provided for unusual values
**Validates: Requirements 8.2**

### Property 38: Configuration Snapshot Preservation
*For any* experiment execution, configuration snapshots should be saved with results for reproducibility
**Validates: Requirements 8.3**

### Property 39: Hardware-Adaptive Configuration
*For any* hardware change, capabilities should be automatically detected and defaults should be adjusted accordingly
**Validates: Requirements 8.4**

### Property 40: Result Sharing Completeness
*For any* result sharing, all necessary configuration information should be included in output files
**Validates: Requirements 8.5**

### Property 41: DINOv2 Model Loading
*For any* DINOv2 backbone loading, the ViT-L/14 variant should be loaded successfully with correct architecture
**Validates: Requirements 9.1**

### Property 42: DINOv2 Backbone Freezing
*For any* feature extraction operation, the DINOv2 backbone parameters should remain frozen to preserve pre-trained knowledge
**Validates: Requirements 9.2**

### Property 43: 4-Channel Input Adaptation
*For any* RGB + Depth input processing, the DINOv2 model should correctly accept and process 4-channel inputs
**Validates: Requirements 9.3**

### Property 44: Feature Dimensionality Consistency
*For any* feature extraction, the output should be 1024-dimensional feature vectors suitable for downstream tasks
**Validates: Requirements 9.4**

### Property 45: Feature Quality Validation
*For any* backbone validation, feature quality should be verified using visualization and similarity metrics
**Validates: Requirements 9.5**

### Property 46: Multi-Task Architecture Completeness
*For any* multi-task model creation, the model should include three specialized heads for height, direction, and breaking type
**Validates: Requirements 10.1**

### Property 47: Task-Specific Projection Layers
*For any* feature processing, task-specific projection layers should be applied before each prediction head
**Validates: Requirements 10.2**

### Property 48: Loss Function Balancing
*For any* multi-task training, weighted loss functions should prevent task dominance and enable balanced learning
**Validates: Requirements 10.3**

### Property 49: Simultaneous Prediction Output
*For any* inference operation, the multi-task model should output all three metrics simultaneously with confidence scores
**Validates: Requirements 10.4**

### Property 50: Independent Task Learning
*For any* architecture validation, each head should be capable of learning independently without negative transfer
**Validates: Requirements 10.5**

### Property 51: Synthetic Pre-Training Duration
*For any* pre-training phase, the system should train on synthetic data with perfect labels for 50+ epochs
**Validates: Requirements 11.1**

### Property 52: Real Data Fine-Tuning
*For any* fine-tuning phase, the system should adapt the pre-trained model using limited real beach cam data
**Validates: Requirements 11.2**

### Property 53: Learning Rate Schedule Application
*For any* training phase, cosine annealing with warm restarts should be applied for stable convergence
**Validates: Requirements 11.3**

### Property 54: Per-Task Loss Monitoring
*For any* training monitoring, per-task losses and overall validation performance should be tracked
**Validates: Requirements 11.4**

### Property 55: Best Checkpoint Selection
*For any* training completion, the best model checkpoint should be saved based on combined validation metrics
**Validates: Requirements 11.5**

### Property 56: Model Loading Completeness
*For any* inference initialization, the system should load the best checkpoint with all three prediction heads
**Validates: Requirements 12.1**

### Property 57: Inference Performance Target
*For any* new image processing, the Wave Analyzer should complete depth extraction and inference in under 30 seconds
**Validates: Requirements 12.2**

### Property 58: Wave Metrics Output Format
*For any* inference completion, the system should display wave height (meters and feet), direction, and breaking type
**Validates: Requirements 12.3**

### Property 59: Low Confidence Warning
*For any* low confidence prediction, the system should display confidence scores and warning messages
**Validates: Requirements 12.4**

### Property 60: Result Visualization
*For any* prediction visualization, the system should overlay predictions on the original image with color-coded annotations
**Validates: Requirements 12.5**

### Property 61: Wave Height Evaluation Metrics
*For any* wave height evaluation, the system should compute MAE, RMSE, and R² scores against ground truth
**Validates: Requirements 13.1**

### Property 62: Direction Classification Metrics
*For any* direction evaluation, the system should compute accuracy, precision, recall, and F1 scores for each class
**Validates: Requirements 13.2**

### Property 63: Breaking Type Confusion Matrix
*For any* breaking type evaluation, the system should compute confusion matrices and per-class accuracy metrics
**Validates: Requirements 13.3**

### Property 64: Sim-to-Real Transfer Gap Quantification
*For any* performance comparison, the system should quantify the sim-to-real transfer gap
**Validates: Requirements 13.4**

### Property 65: Comprehensive Evaluation Report
*For any* evaluation completion, the system should create comprehensive visualizations and summary statistics
**Validates: Requirements 13.5**

### Property 66: Beach Cam Resolution Support
*For any* beach cam image input, the Wave Analyzer should accept images with resolution between 480p and 4K
**Validates: Requirements 14.1**

### Property 67: Image Quality Enhancement
*For any* poor quality or low contrast image, the Wave Analyzer should enhance image quality before processing
**Validates: Requirements 14.2**

### Property 68: Multi-Format Support
*For any* image format input, the Wave Analyzer should support JPEG, PNG, and WebP formats
**Validates: Requirements 14.3**

### Property 69: Ocean Region Detection
*For any* image containing non-ocean areas, the Wave Analyzer should automatically detect and focus on ocean regions
**Validates: Requirements 14.4**

### Property 70: Quality Failure Feedback
*For any* processing failure due to image quality, the Wave Analyzer should provide specific feedback about quality issues
**Validates: Requirements 14.5**

### Property 71: Sharp Wave Edge Preservation
*For any* beach cam image processing, Depth-Anything-V2 should generate depth maps that preserve sharp wave edges
**Validates: Requirements 15.1**

### Property 72: Water Texture Capture
*For any* water texture presence, Depth-Anything-V2 should capture fine-grained water surface details
**Validates: Requirements 15.2**

### Property 73: Far-Field Depth Sensitivity
*For any* distant wave visibility, Depth-Anything-V2 should maintain depth sensitivity for far-field objects
**Validates: Requirements 15.3**

### Property 74: Depth Normalization for Wave Prominence
*For any* depth map generation, the system should normalize depth values so waves stand out against ocean surface
**Validates: Requirements 15.4**

### Property 75: Depth Quality Statistical Validation
*For any* depth extraction completion, the system should validate depth map quality using statistical measures
**Validates: Requirements 15.5**

### Property 76: Wave Height Measurement Accuracy
*For any* wave analysis, the Wave Analyzer should measure wave height with ±0.2m accuracy
**Validates: Requirements 16.1**

### Property 77: Dominant Wave Height Reporting
*For any* multiple wave presence, the Wave Analyzer should report the dominant wave height
**Validates: Requirements 16.2**

### Property 78: Dual Unit Height Measurement
*For any* wave height measurement, the Wave Analyzer should provide measurements in both meters and feet
**Validates: Requirements 16.3**

### Property 79: Extreme Condition Flagging
*For any* wave height below 0.5m or above 8.0m, the Wave Analyzer should flag extreme conditions
**Validates: Requirements 16.4**

### Property 80: Height Confidence Scoring
*For any* low measurement confidence, the Wave Analyzer should include confidence scores with height estimates
**Validates: Requirements 16.5**

### Property 81: Direction Classification Accuracy
*For any* wave direction analysis, the Wave Analyzer should classify direction as Left, Right, or Straight with 90% accuracy
**Validates: Requirements 17.1, 17.2**

### Property 82: Dominant Direction Reporting
*For any* varying wave direction, the Wave Analyzer should report the dominant direction and note mixed conditions
**Validates: Requirements 17.3**

### Property 83: Primary Direction Pattern Identification
*For any* multiple wave train presence, the Wave Analyzer should identify the primary direction pattern
**Validates: Requirements 17.4**

### Property 84: Direction Confidence Scoring
*For any* uncertain direction analysis, the Wave Analyzer should provide confidence scores for each direction category
**Validates: Requirements 17.5**

### Property 85: Breaking Type Classification Accuracy
*For any* breaking type classification, the Wave Analyzer should distinguish between Spilling, Plunging, and Surging with 92% accuracy
**Validates: Requirements 18.1, 18.2**

### Property 86: Breaking Type Distribution Reporting
*For any* multiple breaking type presence, the Wave Analyzer should report the dominant type and percentage breakdown
**Validates: Requirements 18.3**

### Property 87: Breaking Type Confidence Scoring
*For any* unclear breaking pattern, the Wave Analyzer should provide classification confidence scores
**Validates: Requirements 18.4**

### Property 88: No Breaking Detection
*For any* absence of clear breaking, the Wave Analyzer should report "No Breaking" with appropriate reasoning
**Validates: Requirements 18.5**

## Wave Analysis Design Decisions

### 1. DINOv2 as Feature Extraction Backbone

**Decision**: Use DINOv2 ViT-L/14 as the frozen feature extraction backbone instead of training a custom CNN from scratch.

**Rationale**:
- DINOv2's self-supervised learning on diverse visual data provides superior geometric understanding
- Pre-trained on 142M images, capturing universal visual patterns including water and wave geometry
- Vision Transformer architecture excels at capturing long-range spatial relationships critical for wave analysis
- Frozen backbone prevents overfitting on limited real beach cam data
- 1024-dimensional features provide rich representation for downstream tasks

**Trade-offs**:
- Larger model size (300M parameters) requires more memory
- Slower inference compared to lightweight CNNs
- Requires 4-channel input adaptation for RGB + Depth

### 2. Multi-Task Learning Architecture

**Decision**: Implement a single multi-task model with three specialized heads instead of three separate models.

**Rationale**:
- Shared feature extraction reduces computational cost (3x faster than separate models)
- Multi-task learning improves generalization through shared representations
- Task-specific projection layers prevent negative transfer between tasks
- Weighted loss functions enable balanced learning across tasks
- Single model simplifies deployment and maintenance

**Trade-offs**:
- More complex training dynamics requiring careful loss balancing
- Potential for negative transfer if tasks conflict
- Harder to debug individual task performance

### 3. Sim-to-Real Training Strategy

**Decision**: Pre-train on synthetic data with perfect labels, then fine-tune on limited real data.

**Rationale**:
- Solves the manual labeling challenge (labeling wave metrics is time-consuming and subjective)
- FLUX.1-dev + ControlNet-Depth generates photorealistic synthetic images with automatic labels
- Pre-training establishes strong feature representations and task understanding
- Fine-tuning adapts to real-world beach cam characteristics (lighting, weather, camera angles)
- Enables training with 1000+ synthetic images and only 100-200 real labeled images

**Trade-offs**:
- Sim-to-real gap may limit final performance
- Requires careful domain randomization in synthetic generation
- Two-phase training increases overall training time

### 4. Depth-Anything-V2 for Depth Extraction

**Decision**: Use Depth-Anything-V2-Large instead of MiDaS or other depth estimation models.

**Rationale**:
- Superior edge preservation critical for capturing sharp wave boundaries
- Better far-field depth sensitivity for distant waves
- Captures fine-grained water surface texture
- Trained on diverse datasets including water scenes
- Robust to varying lighting and weather conditions

**Trade-offs**:
- Larger model (335M parameters) requires more GPU memory
- Slower processing compared to smaller depth models
- May require depth normalization for optimal wave prominence

### 5. FLUX.1-dev for Synthetic Generation

**Decision**: Use FLUX.1-dev + ControlNet-Depth instead of Stable Diffusion XL or other generative models.

**Rationale**:
- State-of-the-art photorealism for water and wave generation
- Superior depth conditioning for physics-accurate wave geometry
- Better prompt following for precise wave characteristic control
- Generates diverse conditions (weather, lighting, wave types)
- Produces high-quality training data that transfers well to real images

**Trade-offs**:
- Requires significant GPU memory (24GB+ recommended)
- Slower generation compared to smaller models
- More complex prompt engineering for optimal results

### 6. Confidence Scoring for All Predictions

**Decision**: Include confidence scores with all wave metric predictions.

**Rationale**:
- Enables users to assess prediction reliability
- Critical for safety-sensitive surfing decisions
- Allows filtering of low-confidence predictions
- Provides feedback for model improvement
- Supports active learning for efficient data labeling

**Trade-offs**:
- Adds complexity to model output
- Requires calibration to ensure confidence accuracy
- May confuse users if not presented clearly

## Error Handling

### Error Categories and Strategies

1. **Data Errors**: Missing files, corrupted images, format issues
   - Strategy: Validation before processing, graceful skipping, detailed logging
   - Recovery: Continue with valid data, provide repair suggestions

2. **Memory Errors**: GPU OOM, excessive RAM usage
   - Strategy: Dynamic batch sizing, automatic fallback to CPU
   - Recovery: Reduce batch size, clear memory, restart with smaller batches

3. **Network Errors**: Model download failures, connection timeouts
   - Strategy: Exponential backoff retry, alternative download sources
   - Recovery: Retry with different endpoints, use cached models if available

4. **Processing Errors**: Model inference failures, computation errors
   - Strategy: Individual item error handling, partial result preservation
   - Recovery: Skip failed items, save partial results, provide recovery instructions

5. **Configuration Errors**: Invalid parameters, missing configuration
   - Strategy: Validation with clear error messages, sensible defaults
   - Recovery: Use default values, guide user to fix configuration

## Testing Strategy

### Dual Testing Approach

The testing strategy combines unit tests for specific functionality with property-based tests for comprehensive validation:

**Unit Tests**: Focus on specific examples, edge cases, and error conditions
- Test individual utility functions with known inputs and expected outputs
- Validate error handling with specific failure scenarios
- Test configuration loading with various valid and invalid configurations
- Verify memory cleanup with controlled memory allocation scenarios
- Test wave analysis components with known beach cam images and expected metrics
- Validate DINOv2 feature extraction with sample inputs
- Test multi-task model predictions with synthetic data

**Property-Based Tests**: Verify universal properties across all inputs
- Test data format consistency across different data types and sizes
- Validate memory management across various memory conditions
- Test error handling across different failure modes
- Verify progress tracking across different processing scenarios
- Test wave height prediction accuracy across diverse wave conditions
- Validate direction classification across different wave angles
- Test breaking type classification across various wave patterns
- Verify sim-to-real transfer across synthetic and real data distributions

### Property Test Configuration

Each property test should run a minimum of 100 iterations to ensure comprehensive coverage through randomization. Tests will be tagged with the format: **Feature: swellsight-pipeline-improvements, Property {number}: {property_text}**

### Testing Framework

The testing will use pytest for unit tests and Hypothesis for property-based testing, integrated directly into the notebook cells for immediate validation during development.

Example test structure:
```python
# Unit test example
def test_save_load_consistency():
    test_data = {"images": ["img1.jpg"], "scores": [0.85]}
    save_stage_results(test_data, "test_stage", {})
    loaded_data = load_previous_results("test_stage", ["test_stage.json"])
    assert loaded_data == test_data

# Property test example  
@given(st.lists(st.text(), min_size=1), st.floats(0.0, 1.0))
def test_batch_size_adaptation(image_list, memory_fraction):
    """Feature: swellsight-pipeline-improvements, Property 6: Dynamic batch sizing adaptation"""
    available_memory = 8 * 1024**3 * memory_fraction  # Simulate different memory levels
    batch_size = get_optimal_batch_size(available_memory, estimate_image_memory_usage())
    assert batch_size > 0
    assert batch_size <= len(image_list)
    assert batch_size * estimate_image_memory_usage() <= available_memory

# Wave analysis property test example
@given(st.floats(0.5, 8.0), st.sampled_from(["Left", "Right", "Straight"]))
def test_wave_height_prediction(true_height, true_direction):
    """Feature: swellsight-pipeline-improvements, Property 76: Wave height measurement accuracy"""
    # Generate synthetic image with known wave parameters
    synthetic_image = generate_synthetic_wave(height=true_height, direction=true_direction)
    
    # Predict wave metrics
    predictions = wave_analyzer.predict(synthetic_image)
    
    # Verify accuracy within tolerance
    assert abs(predictions["wave_height"]["meters"] - true_height) <= 0.2
    assert predictions["direction"]["class"] == true_direction
```