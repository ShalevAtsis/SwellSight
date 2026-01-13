#!/usr/bin/env python3
"""
Quick fix for the notebook import issue
Run this script to add fallback functions for missing utilities
"""

import sys
import time
from pathlib import Path

# Define fallback functions that can be used directly in the notebook
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

# Make functions available globally
globals().update({
    'create_progress_bar': create_progress_bar,
    'display_stage_summary': display_stage_summary,
    'retry_with_backoff': retry_with_backoff,
    'monitor_memory': monitor_memory,
    'get_optimal_batch_size': get_optimal_batch_size,
    'cleanup_variables': cleanup_variables,
    'save_stage_results': save_stage_results,
    'load_config': load_config,
    'validate_config': validate_config,
    'BasicProgressBar': BasicProgressBar
})

if __name__ == "__main__":
    print("✅ Fallback functions defined successfully!")
    print("You can now run the notebook cells that were failing.")
    
    # Test the progress bar
    print("\n🧪 Testing progress bar...")
    pbar = create_progress_bar(5, "Testing")
    for i in range(5):
        time.sleep(0.5)
        pbar.update(1)
    pbar.close()
    print("✅ Progress bar test completed!")