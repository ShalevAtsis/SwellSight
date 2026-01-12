"""
Example: How to integrate SwellSight utilities into Jupyter notebooks
This shows the code cells you would add to your existing notebooks
"""

# =============================================================================
# CELL 1: Import utilities (Add this to the beginning of each notebook)
# =============================================================================

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

print("✓ SwellSight utilities loaded successfully")

# =============================================================================
# CELL 2: Load configuration (Add this after imports)
# =============================================================================

# Load pipeline configuration
config = load_config("config.json")
print(f"✓ Configuration loaded: {config['pipeline']['name']} v{config['pipeline']['version']}")

# Extract commonly used settings
batch_size = config['processing']['batch_size']
quality_threshold = config['processing']['quality_threshold']
data_dir = Path(config['paths']['data_dir'])
output_dir = Path(config['paths']['output_dir'])

print(f"✓ Data directory: {data_dir}")
print(f"✓ Output directory: {output_dir}")
print(f"✓ Quality threshold: {quality_threshold}")

# =============================================================================
# CELL 3: Check dependencies (Add this to verify previous stages)
# =============================================================================

# Check if previous stages are completed (example for notebook 03)
stage_name = "depth_extraction"  # Change this for each notebook
dep_status = check_dependencies(stage_name)

if not dep_status['all_satisfied']:
    print("⚠️ Missing dependencies:")
    for missing in dep_status['missing_dependencies']:
        print(f"  - {missing}")
    print("Please complete previous stages first.")
else:
    print("✓ All dependencies satisfied")

# =============================================================================
# CELL 4: Memory optimization (Add this before processing loops)
# =============================================================================

# Get optimal batch size based on available memory
if batch_size == "auto":
    optimal_batch_size = get_optimal_batch_size(max_batch_size=32)
    print(f"✓ Calculated optimal batch size: {optimal_batch_size}")
else:
    optimal_batch_size = batch_size
    print(f"✓ Using configured batch size: {optimal_batch_size}")

# Monitor initial memory usage
memory_info = monitor_memory()
print(f"✓ System memory: {memory_info.get('system_percent', 0):.1f}% used")
if 'gpu_percent' in memory_info:
    print(f"✓ GPU memory: {memory_info.get('gpu_percent', 0):.1f}% used")

# =============================================================================
# CELL 5: Processing with progress tracking (Replace your processing loops)
# =============================================================================

# Example: Processing images with utilities
def process_images_with_utilities(image_paths, stage_name="processing"):
    """Example of how to process images using the utilities"""
    
    # Create progress bar
    progress_bar = create_progress_bar(len(image_paths), f"Processing {stage_name}")
    
    processed_results = []
    failed_images = []
    
    try:
        for i, image_path in enumerate(image_paths):
            try:
                # Validate image quality first
                quality_result = validate_image_quality(image_path, quality_threshold)
                
                if not quality_result['valid']:
                    print(f"⚠️ Skipping low quality image: {image_path}")
                    failed_images.append({
                        'path': image_path,
                        'reason': 'Low quality',
                        'issues': quality_result['issues']
                    })
                    continue
                
                # Process the image (your existing processing code here)
                # result = your_processing_function(image_path)
                result = f"processed_{Path(image_path).name}"  # Placeholder
                
                processed_results.append({
                    'input_path': image_path,
                    'output_path': result,
                    'quality_score': quality_result['score']
                })
                
                # Update progress with memory info
                current_memory = monitor_memory()
                memory_info_str = f"Mem: {current_memory.get('system_percent', 0):.0f}%"
                progress_bar.update(1)
                
            except Exception as e:
                print(f"✗ Error processing {image_path}: {e}")
                failed_images.append({
                    'path': image_path,
                    'reason': str(e)
                })
        
        progress_bar.close()
        
        # Display stage summary
        stage_metrics = {
            'total_images': len(image_paths),
            'processed_successfully': len(processed_results),
            'failed_images': len(failed_images),
            'success_rate': len(processed_results) / len(image_paths) if image_paths else 0,
            'average_quality_score': sum(r['quality_score'] for r in processed_results) / len(processed_results) if processed_results else 0
        }
        
        display_stage_summary(stage_name, stage_metrics)
        
        return processed_results, failed_images
        
    except Exception as e:
        progress_bar.close()
        print(f"✗ Processing failed: {e}")
        return processed_results, failed_images

# =============================================================================
# CELL 6: Error handling with retry (Wrap risky operations)
# =============================================================================

# Example: Download model with retry logic
def download_model_with_retry(model_name, download_function):
    """Download model with automatic retry on failure"""
    
    def download_attempt():
        return download_function(model_name)
    
    try:
        model = retry_with_backoff(download_attempt, max_retries=3)
        print(f"✓ Model downloaded successfully: {model_name}")
        return model
    except Exception as e:
        print(f"✗ Failed to download model after retries: {e}")
        return None

# =============================================================================
# CELL 7: Save stage results (Add this at the end of each notebook)
# =============================================================================

# Save results for next stage
stage_results = {
    'processed_images': [r['output_path'] for r in processed_results],
    'quality_scores': [r['quality_score'] for r in processed_results],
    'failed_images': failed_images,
    'processing_metrics': stage_metrics
}

stage_metadata = {
    'processing_time_seconds': stage_metrics.get('duration_seconds', 0),
    'model_used': config['models']['depth_model'],  # Example
    'batch_size_used': optimal_batch_size,
    'quality_threshold': quality_threshold
}

# Save results
success = save_stage_results(stage_results, stage_name, stage_metadata)
if success:
    print(f"✓ Stage results saved for: {stage_name}")
else:
    print(f"✗ Failed to save stage results for: {stage_name}")

# =============================================================================
# CELL 8: Cleanup (Add this at the end of each notebook)
# =============================================================================

# Clean up large variables to free memory
large_variables = [
    # Add your large variables here, e.g.:
    # model, processed_images, depth_maps, etc.
]

cleanup_variables(large_variables)
print("✓ Memory cleanup completed")

# Final memory status
final_memory = monitor_memory()
print(f"✓ Final memory usage: {final_memory.get('system_percent', 0):.1f}% system")
if 'gpu_percent' in final_memory:
    print(f"✓ Final GPU usage: {final_memory.get('gpu_percent', 0):.1f}%")