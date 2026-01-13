# Add this cell to your notebook before the failing cell to define fallback functions

import sys
import time
from pathlib import Path

# Define fallback functions for when utils import fails
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
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb > 12:
                return min(max_batch_size, 16)
            elif gpu_memory_gb > 8:
                return min(max_batch_size, 8)
            else:
                return min(max_batch_size, 4)
        else:
            return min(max_batch_size, 2)
    except ImportError:
        return min(max_batch_size, 4)

def cleanup_variables(variables):
    """Basic cleanup"""
    import gc
    gc.collect()

def save_stage_results(results, stage_name, metadata=None):
    """Basic results saving"""
    import json
    from pathlib import Path
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / f"{stage_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return True

def load_config(config_path):
    """Basic config loading"""
    import json
    with open(config_path, 'r') as f:
        return json.load(f)

class BasicProgressBar:
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
            print()
    
    def close(self):
        pass

print("✅ Fallback functions loaded successfully!")