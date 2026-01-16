#!/usr/bin/env python3
"""
SwellSight Training Readiness Checker

This script checks if your system is ready for training the SwellSight model.
It verifies hardware, software, data, and disk space requirements.

Usage:
    python scripts/check_training_readiness.py
"""

import sys
import subprocess
from pathlib import Path
import shutil

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_check(name, status, details=""):
    """Print a check result."""
    symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{symbol} {name:.<50} {status_text}{reset}")
    if details:
        print(f"  → {details}")

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 8)
    
    is_ok = version >= required
    details = f"Found Python {version.major}.{version.minor}.{version.micro}"
    if not is_ok:
        details += f" (Need Python {required[0]}.{required[1]}+)"
    
    return is_ok, details

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            details = f"{gpu_name} with {gpu_memory:.1f}GB VRAM"
            
            # Check if enough VRAM
            if gpu_memory < 6:
                details += " (⚠️  Recommended: 8GB+ for training)"
        else:
            details = "No GPU detected (Training will be VERY slow on CPU)"
        
        return has_cuda, details
    except ImportError:
        return False, "PyTorch not installed"

def check_disk_space():
    """Check available disk space."""
    try:
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024**3)
        
        required_gb = 50
        is_ok = free_gb >= required_gb
        
        details = f"{free_gb:.1f}GB free"
        if not is_ok:
            details += f" (Need {required_gb}GB+)"
        
        return is_ok, details
    except Exception as e:
        return False, f"Could not check disk space: {e}"

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers',
        'diffusers',
        'opencv-python',
        'numpy',
        'pillow',
        'tqdm',
        'pyyaml'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    is_ok = len(missing) == 0
    
    if is_ok:
        details = f"All {len(required_packages)} required packages installed"
    else:
        details = f"Missing: {', '.join(missing)}"
    
    return is_ok, details

def check_directory_structure():
    """Check if required directories exist."""
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/depth_maps',
        'data/synthetic',
        'data/augmented',
        'models/checkpoints',
        'outputs/logs'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    is_ok = len(missing) == 0
    
    if is_ok:
        details = "All required directories exist"
    else:
        details = f"Missing: {', '.join(missing[:3])}"
        if len(missing) > 3:
            details += f" and {len(missing)-3} more"
    
    return is_ok, details

def check_data_availability():
    """Check if training data is available."""
    checks = {}
    
    # Check raw images
    raw_dir = Path('data/raw/beach_cams')
    if raw_dir.exists():
        raw_images = list(raw_dir.glob('*.jpg')) + list(raw_dir.glob('*.png'))
        checks['raw'] = (len(raw_images), len(raw_images) >= 100)
    else:
        checks['raw'] = (0, False)
    
    # Check depth maps
    depth_dir = Path('data/depth_maps')
    if depth_dir.exists():
        depth_maps = list(depth_dir.glob('*_depth.npy'))
        checks['depth'] = (len(depth_maps), len(depth_maps) >= 100)
    else:
        checks['depth'] = (0, False)
    
    # Check synthetic data
    synth_dir = Path('data/synthetic')
    if synth_dir.exists():
        synth_images = [f for f in synth_dir.glob('*.npy') if '_labels' not in f.name]
        checks['synthetic'] = (len(synth_images), len(synth_images) >= 500)
    else:
        checks['synthetic'] = (0, False)
    
    return checks

def check_training_config():
    """Check if training configuration exists."""
    config_path = Path('configs/training_config.yaml')
    
    is_ok = config_path.exists()
    details = "Configuration file found" if is_ok else "Configuration file missing"
    
    return is_ok, details

def main():
    """Run all checks."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "SwellSight Training Readiness Check" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    
    all_checks = []
    
    # System Requirements
    print_header("System Requirements")
    
    check, details = check_python_version()
    print_check("Python Version", check, details)
    all_checks.append(check)
    
    check, details = check_gpu()
    print_check("GPU Availability", check, details)
    # GPU is recommended but not required
    
    check, details = check_disk_space()
    print_check("Disk Space", check, details)
    all_checks.append(check)
    
    # Software Dependencies
    print_header("Software Dependencies")
    
    check, details = check_dependencies()
    print_check("Required Packages", check, details)
    all_checks.append(check)
    
    # Directory Structure
    print_header("Directory Structure")
    
    check, details = check_directory_structure()
    print_check("Required Directories", check, details)
    if not check:
        print("  💡 Run: mkdir -p data/{raw,processed,depth_maps,synthetic,augmented} models/checkpoints outputs/logs")
    
    # Data Availability
    print_header("Training Data")
    
    data_checks = check_data_availability()
    
    raw_count, raw_ok = data_checks['raw']
    print_check("Raw Beach Cam Images", raw_ok, 
               f"{raw_count} images (Need 100+)")
    
    depth_count, depth_ok = data_checks['depth']
    print_check("Depth Maps", depth_ok, 
               f"{depth_count} depth maps (Need 100+)")
    
    synth_count, synth_ok = data_checks['synthetic']
    print_check("Synthetic Images", synth_ok, 
               f"{synth_count} images (Need 500+)")
    
    # Configuration
    print_header("Configuration")
    
    check, details = check_training_config()
    print_check("Training Config", check, details)
    
    # Summary
    print_header("Summary")
    
    system_ready = all(all_checks)
    data_ready = raw_ok and depth_ok and synth_ok
    
    if system_ready and data_ready:
        print("\n✅ Your system is READY for training!")
        print("\nNext steps:")
        print("  1. Review training configuration: configs/training_config.yaml")
        print("  2. Start training: python scripts/train_model.py")
        print("  3. Monitor progress in outputs/logs/")
    elif system_ready and not data_ready:
        print("\n⚠️  System is ready, but data preparation needed!")
        print("\nNext steps:")
        if not raw_ok:
            print("  1. Collect beach cam images → data/raw/beach_cams/")
            print("     Need at least 100 images")
        if not depth_ok:
            print("  2. Extract depth maps: python scripts/extract_depth_maps.py")
        if not synth_ok:
            print("  3. Generate synthetic data: python scripts/generate_synthetic_data.py")
            print("     This will take 4-8 hours!")
        print("\n📖 See: docs/TRAINING_FROM_SCRATCH.md for detailed guide")
    else:
        print("\n❌ System is NOT ready for training")
        print("\nIssues to fix:")
        if not all_checks:
            print("  • Install missing dependencies: pip install -r requirements/training.txt")
        if not check_directory_structure()[0]:
            print("  • Create directories: mkdir -p data/{raw,processed,depth_maps,synthetic,augmented}")
        print("\n📖 See: docs/TRAINING_FROM_SCRATCH.md for setup guide")
    
    print("\n" + "="*70 + "\n")
    
    # Return exit code
    return 0 if system_ready else 1

if __name__ == "__main__":
    sys.exit(main())
