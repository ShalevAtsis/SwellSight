import sys
import os
# Add the BASE directory to sys.path to allow importing modules from there
sys.path.insert(0, BASE)

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from src.swellsight.core.depth_extractor import DepthAnythingV2Extractor, ProcessingError

# Set HF_TOKEN environment variable from the provided GITHUB_TOKEN
# This assumes GITHUB_TOKEN can be used for Hugging Face authentication
if 'token' in locals() and token is not None:
    os.environ['HF_TOKEN'] = token
elif os.getenv('HF_TOKEN') is None:
    print("WARNING: HF_TOKEN environment variable not set. Model loading might fail.")

def extract_depth_maps(input_dir, output_dir, use_gpu=True):
    """Extract depth maps from beach cam images."""

    print("Initializing Depth-Anything-V2...")
    # Force fp32 precision to ensure OpenCV compatibility (cv2.resize does not support float16)
    extractor = DepthAnythingV2Extractor(
        model_size="large",
        precision="fp32",
        enable_optimization=True
    )
    print("✓ Depth extractor initialized")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    print(f"\nExtracting depth maps for {len(images)} images...")

    successful_extractions = 0
    for img_path in tqdm(images):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping {img_path.name}: Could not read image.")
            continue

        # Add a check for image dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            print(f"Skipping {img_path.name}: Image has zero dimension(s). Shape: {img.shape}")
            continue

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            # Extract depth
            depth_map = extractor.extract_depth(img_rgb)
        except ProcessingError as e:
            print(f"Skipping {img_path.name}: Depth extraction failed with error: {e}")
            continue

        # Handle tuple return (depth_map, performance_metrics)
        if isinstance(depth_map, tuple):
            depth_map = depth_map[0]

        # Save depth map
        output_file = output_path / f"{img_path.stem}_depth.npy"
        np.save(output_file, depth_map.data)

        # Also save visualization
        vis_file = output_path / f"{img_path.stem}_depth_vis.jpg"
        # Normalize to 0-255 for visualization
        depth_norm = cv2.normalize(depth_map.data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(str(vis_file), depth_norm)
        successful_extractions += 1

    print(f"✓ Successfully extracted {successful_extractions} depth maps out of {len(images)} images")

if __name__ == "__main__":
    extract_depth_maps(
        f'{DATA_DIR}/processed',
        f'{DATA_DIR}/depth_maps',
        use_gpu=True
    )