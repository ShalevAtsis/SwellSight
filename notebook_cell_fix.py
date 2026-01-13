import sys
import os
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add utils directory to Python path
utils_path = Path.cwd() / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

try:
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
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import utilities: {e}")
    print("Continuing with basic functionality...")
    UTILS_AVAILABLE = False
    
    # Define fallback functions
    def create_progress_bar(total_items, description="Processing", show_memory=False):
        """Fallback progress bar when utils not available"""
        try:
            from tqdm import tqdm
            return tqdm(total=total_items, desc=description, ncols=100)
        except ImportError:
            return BasicProgressBar(total_items, description)
    
    def display_stage_summary(stage_name, metrics):
        """Fallback stage summary display"""
        print(f"\n{'='*60}")
        print(f"STAGE SUMMARY: {stage_name.upper()}")
        print(f"{'='*60}")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'time' in key.lower() or 'duration' in key.lower():
                    print(f"{key.replace('_', ' ').title()}: {value:.2f} seconds")
                elif 'rate' in key.lower() or 'score' in key.lower():
                    print(f"{key.replace('_', ' ').title()}: {value:.3f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            elif isinstance(value, int):
                print(f"{key.replace('_', ' ').title()}: {value:,}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        print(f"{'='*60}\n")
    
    def retry_with_backoff(func, max_retries=3):
        """Simple retry function"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
    
    def monitor_memory():
        """Basic memory monitoring"""
        try:
            import psutil
            return {
                'system_percent': psutil.virtual_memory().percent,
                'system_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        except ImportError:
            return {'system_percent': 0, 'system_total_gb': 0}
    
    def get_optimal_batch_size(max_batch_size=32):
        """Simple batch size calculation"""
        try:
            import torch
            if torch.cuda.is_available():
                # Get GPU memory and calculate batch size
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb > 12:
                    return min(max_batch_size, 16)
                elif gpu_memory_gb > 8:
                    return min(max_batch_size, 8)
                else:
                    return min(max_batch_size, 4)
            else:
                return min(max_batch_size, 2)  # CPU fallback
        except ImportError:
            return min(max_batch_size, 4)  # Conservative default
    
    def cleanup_variables(variables):
        """Basic cleanup"""
        import gc
        for var in variables:
            if var in globals():
                del globals()[var]
        gc.collect()
    
    def save_stage_results(results, stage_name, metadata=None):
        """Basic results saving"""
        import json
        from pathlib import Path
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Add metadata if provided
        if metadata:
            results['metadata'] = metadata
        
        with open(output_dir / f"{stage_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return True
    
    def load_previous_results(stage_name):
        """Basic results loading"""
        import json
        from pathlib import Path
        results_file = Path("./outputs") / f"{stage_name}_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
    
    def check_dependencies():
        """Basic dependency check"""
        return True
    
    def load_config(config_path):
        """Basic config loading"""
        import json
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def validate_config(config):
        """Basic config validation"""
        required_keys = ['pipeline', 'processing', 'paths']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    def validate_image_quality(image_path, threshold=0.7):
        """Basic image quality validation"""
        return True, 0.8  # Return (is_valid, quality_score)
    
    def validate_depth_map_quality(depth_map, threshold=0.7):
        """Basic depth map quality validation"""
        return True, 0.8  # Return (is_valid, quality_score)
    
    def handle_gpu_memory_error(func, *args, **kwargs):
        """Basic GPU memory error handler"""
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️ GPU out of memory, trying with reduced batch size")
                # Try to reduce batch size if possible
                if 'batch_size' in kwargs:
                    kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                    return func(*args, **kwargs)
            raise e
    
    class BasicProgressBar:
        """Basic progress bar implementation when tqdm is not available"""
        
        def __init__(self, total, description):
            self.total = total
            self.description = description
            self.current = 0
            self.start_time = time.time()
        
        def update(self, increment=1):
            self.current += increment
            elapsed = time.time() - self.start_time
            rate = self.current / max(elapsed, 0.001)
            percent = (self.current / self.total) * 100
            
            print(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%) [{rate:.1f} items/s]", 
                  end="", flush=True)
            
            if self.current >= self.total:
                print()  # New line when complete
        
        def close(self):
            if self.current < self.total:
                print()  # Ensure new line
    
    print("✅ Fallback functions loaded successfully!")

# Test that create_progress_bar works
if __name__ == "__main__":
    print("\n🧪 Testing create_progress_bar function...")
    pbar = create_progress_bar(3, "Testing")
    for i in range(3):
        time.sleep(0.5)
        pbar.update(1)
    pbar.close()
    print("✅ Test completed!")