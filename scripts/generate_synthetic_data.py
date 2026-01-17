import sys
import os
import gc
import torch
from google.colab import userdata
from huggingface_hub import login

# --- 1. SETUP PATHS & IMPORTS (Restart-Safe) ---
# Define BASE if not already present, so this cell runs after a restart
if 'BASE' not in globals():
    BASE = "/content/drive/MyDrive/SwellSight_Colab"
    print(f"✓ BASE path set to: {BASE}")

# Add the BASE directory to sys.path to allow importing modules
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.swellsight.core.synthetic_generator import (
    FLUXControlNetGenerator,
    WeatherConditions,
    GenerationConfig
)

# --- 2. MEMORY CHECK ---
print("\n🔍 Checking GPU Memory...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    free_gb = free_mem / 1024**3
    print(f"   VRAM Free: {free_gb:.2f} GB / {total_mem/1024**3:.2f} GB")

    if free_gb < 10.0:
        print("\n⚠️  WARNING: Low VRAM detected (<10GB).")
        print("   FLUX requires ~12GB. If this fails, RESTART THE RUNTIME (Runtime > Restart session).")
else:
    print("❌ No GPU detected!")

# --------------------------------

def setup_huggingface_auth():
    """Setup authentication for Hugging Face gated models."""
    try:
        hf_token = userdata.get('HF_TOKEN')
        if hf_token:
            print("\n✓ Found HF_TOKEN in secrets, logging in...")
            login(token=hf_token, add_to_git_credential=True)
            os.environ['HF_TOKEN'] = hf_token
            return True
        else:
            print("⚠️ HF_TOKEN not found in Colab secrets.")
            return False
    except Exception as e:
        print(f"⚠️ Could not retrieve HF_TOKEN: {e}")
        return False

def generate_synthetic_dataset(depth_dir, output_dir, num_images=500):
    """Generate synthetic wave images from depth maps."""

    if not setup_huggingface_auth():
        print("❌ Authentication failed. Please check HF_TOKEN.")
        return

    # Initialize generator
    print("\n🚀 Initializing FLUX ControlNet Generator...")
    try:
        generator = FLUXControlNetGenerator()
        print("✓ Generator initialized")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Failed to initialize generator.\nError: {e}")
        if "out of memory" in str(e).lower():
            print("\n🛑 DIAGNOSIS: GPU Out of Memory.")
            print("👉 FIX: Restart the Runtime (Runtime > Restart session) and run ONLY this cell.")
        return

    # Setup directories
    if 'DATA_DIR' not in globals():
        DATA_DIR = f"{BASE}/data"

    # Use directory arguments if provided, else fall back to constructed paths
    # Note: The function args 'depth_dir' passed in __main__ are used here
    depth_path = Path(depth_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load depth maps
    depth_files = list(depth_path.glob('*_depth.npy'))
    print(f"Found {len(depth_files)} depth maps in {depth_path}")

    if len(depth_files) == 0:
        print("❌ No depth maps found! Run depth extraction first (or check path).")
        return

    print(f"\n🎨 Generating {num_images} synthetic images...")
    print("⏳ Estimated time: ~2-3 hours on T4 GPU")

    try:
        labeled_dataset = generator.create_balanced_dataset(target_size=num_images)
    except Exception as e:
        print(f"❌ Generation loop failed: {e}")
        return

    generated_count = len(labeled_dataset.images)
    print(f"\n✓ Generation loop complete. Created {generated_count} images.")

    if generated_count == 0:
        print("⚠️ No images were generated. Check logs above for specific errors.")
        return

    # Save results
    print("Saving dataset to disk...")
    for i, synthetic_image in enumerate(tqdm(labeled_dataset.images)):
        img_file = output_path / f"synthetic_{i:04d}.npy"
        np.save(img_file, synthetic_image.rgb_data)

        label_file = output_path / f"synthetic_{i:04d}_labels.npy"
        labels = {
            'height_meters': synthetic_image.ground_truth_labels.height_meters,
            'direction': synthetic_image.ground_truth_labels.direction,
            'breaking_type': synthetic_image.ground_truth_labels.breaking_type,
            'height_confidence': synthetic_image.ground_truth_labels.height_confidence,
            'direction_confidence': synthetic_image.ground_truth_labels.direction_confidence,
            'breaking_confidence': synthetic_image.ground_truth_labels.breaking_confidence
        }
        np.save(label_file, labels)

    print(f"\n✅ Successfully saved {generated_count} synthetic images")

if __name__ == "__main__":
    # Ensure DATA_DIR is defined locally for this block
    local_data_dir = f"{BASE}/data"

    generate_synthetic_dataset(
        f'{local_data_dir}/depth_maps',
        f'{local_data_dir}/synthetic',
        num_images=500
    )