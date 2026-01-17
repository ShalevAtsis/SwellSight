# SwellSight Utilities Integration Guide

## How to Use the Utilities in Your Jupyter Notebooks

The SwellSight utilities are designed to be easily integrated into your existing Jupyter notebooks without requiring you to recreate them as `.ipynb` files. Here are the integration methods:

## Method 1: Direct Import (Recommended)

### Step 1: Add Import Cell
Add this cell at the beginning of each notebook:

```python
import sys
from pathlib import Path

# Add utils directory to Python path
utils_path = Path.cwd() / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

# Import SwellSight utilities
from utils import (
    load_config, validate_config,
    validate_image_quality, validate_depth_map_quality,
    get_optimal_batch_size, cleanup_variables, monitor_memory,
    retry_with_backoff, handle_gpu_memory_error,
    create_progress_bar, display_stage_summary,
    save_stage_results, load_previous_results, check_dependencies
)

print("✅ SwellSight utilities loaded successfully")
```

### Step 2: Load Configuration
Add this cell after imports:

```python
# Load pipeline configuration
config = load_config("config.json")
print(f"✅ Configuration loaded: {config['pipeline']['name']}")

# Extract settings
batch_size = config['processing']['batch_size']
quality_threshold = config['processing']['quality_threshold']
data_dir = Path(config['paths']['data_dir'])
output_dir = Path(config['paths']['output_dir'])
```

### Step 3: Replace Processing Loops
Replace your existing processing loops with utility-enhanced versions:

```python
# Before (your existing code):
for i, image_path in enumerate(image_paths):
    # process image
    result = process_image(image_path)

# After (with utilities):
progress_bar = create_progress_bar(len(image_paths), "Processing images")
for i, image_path in enumerate(image_paths):
    # Validate quality first
    quality_result = validate_image_quality(image_path, quality_threshold)
    if not quality_result['valid']:
        continue
    
    # Process with error handling
    try:
        result = process_image(image_path)
        progress_bar.update(1)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        
progress_bar.close()
```

### Step 4: Add Memory Optimization
Add memory optimization before heavy processing:

```python
# Get optimal batch size
if batch_size == "auto":
    optimal_batch_size = get_optimal_batch_size(max_batch_size=32)
else:
    optimal_batch_size = batch_size

# Monitor memory
memory_info = monitor_memory()
print(f"Memory usage: {memory_info.get('system_percent', 0):.1f}%")
```

### Step 5: Save Results
Add this at the end of each notebook:

```python
# Save stage results
stage_results = {
    'processed_items': processed_results,
    'quality_scores': quality_scores,
    'processing_metrics': metrics
}

success = save_stage_results(stage_results, "stage_name", metadata)
print(f"✅ Results saved: {success}")
```

## Method 2: Copy-Paste Utility Functions

If you prefer to have the utilities directly in your notebooks, you can copy specific functions:

### Example: Copy just the progress tracking
```python
# Cell: Progress Tracking Utility
from tqdm import tqdm

def create_progress_bar(total_items, description="Processing"):
    return tqdm(total=total_items, desc=description, ncols=100)

# Usage in your processing loop
progress_bar = create_progress_bar(len(items), "Processing images")
for item in items:
    # your processing code
    progress_bar.update(1)
progress_bar.close()
```

## Method 3: Magic Commands (Advanced)

You can use Jupyter magic commands to load utilities:

```python
# Cell 1: Load utilities using magic
%run utils/config_manager.py
%run utils/progress_tracker.py

# Cell 2: Use the utilities
config = load_config("config.json")
progress_bar = create_progress_bar(100, "Processing")
```

## Integration Examples by Notebook

### Notebook 01: Setup and Installation
- ✅ Configuration loading and validation
- ✅ Memory monitoring and optimization
- ✅ Error handling for package installation
- ✅ Progress tracking for directory creation

### Notebook 02: Data Import and Preprocessing  
- ✅ Image quality validation
- ✅ Batch processing with dynamic sizing
- ✅ Progress tracking with memory monitoring
- ✅ Error handling for corrupted files

### Notebook 03: Depth Extraction
- ✅ Model loading with retry logic
- ✅ GPU memory error handling with CPU fallback
- ✅ Depth map quality validation
- ✅ Progress tracking with performance metrics

### Notebook 04: Data Augmentation
- ✅ Parameter validation and warnings
- ✅ Memory-aware batch processing
- ✅ Progress tracking for parameter generation

### Notebook 05: Synthetic Generation
- ✅ FLUX model memory optimization
- ✅ Synthetic vs real data comparison
- ✅ Generation quality assessment
- ✅ Error recovery for failed generations

### Notebook 06: Model Training
- ✅ Training progress tracking
- ✅ Checkpoint management
- ✅ Memory optimization for training
- ✅ Error recovery with partial results

### Notebook 07: Data Analysis
- ✅ Statistical analysis utilities
- ✅ Visualization progress tracking
- ✅ Data comparison metrics

### Notebook 08: Evaluation
- ✅ Comprehensive evaluation metrics
- ✅ Result reporting and visualization
- ✅ Performance analysis

## Quick Start Checklist

1. ✅ **Copy the `utils/` directory** to your project root
2. ✅ **Copy `config.json`** to your project root  
3. ✅ **Add import cell** to the beginning of each notebook
4. ✅ **Add configuration loading** after imports
5. ✅ **Replace processing loops** with utility-enhanced versions
6. ✅ **Add memory optimization** before heavy processing
7. ✅ **Add result saving** at the end of each notebook
8. ✅ **Add cleanup** at the end of each notebook

## Benefits of Integration

- 🚀 **Better Performance**: Dynamic batch sizing and memory optimization
- 🛡️ **Error Resilience**: Automatic retry logic and fallback mechanisms  
- 📊 **Progress Tracking**: Visual progress bars with time estimates
- 🔍 **Quality Assurance**: Automatic data quality validation
- 📈 **Monitoring**: Real-time memory and performance monitoring
- 🔄 **Data Flow**: Standardized data exchange between notebooks
- 📋 **Reporting**: Comprehensive stage summaries and metrics

## Troubleshooting

### Import Errors
```python
# If you get import errors, try:
import sys
sys.path.append('./utils')
from config_manager import load_config
```

### Path Issues
```python
# If paths don't work, use absolute paths:
import os
utils_path = os.path.join(os.getcwd(), 'utils')
sys.path.insert(0, utils_path)
```

### Missing Dependencies
```python
# Install missing packages:
!pip install tqdm psutil opencv-python pillow numpy
```

## Need Help?

- 📖 Check the example notebook: `01_Setup_and_Installation_Enhanced.ipynb`
- 🧪 Run the test script: `python test_utils.py`
- 📝 Review the integration example: `notebook_integration_example.py`

The utilities are designed to enhance your existing workflow without requiring major changes to your notebook structure!