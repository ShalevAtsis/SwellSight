#!/usr/bin/env python3
"""
Generate dummy training data for testing the training pipeline.
This creates synthetic .npy files with random data to test the training loop.
"""

import numpy as np
from pathlib import Path
import argparse

def generate_dummy_data(output_dir: str, num_samples: int = 100):
    """Generate dummy training data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} dummy training samples...")
    print(f"Output directory: {output_path}")
    
    # Wave parameters
    directions = ["LEFT", "RIGHT", "STRAIGHT"]
    breaking_types = ["SPILLING", "PLUNGING", "SURGING"]
    
    for i in range(num_samples):
        # Generate random RGB image (224x224x3)
        rgb_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Generate random labels
        labels = {
            'height': np.random.uniform(0.5, 3.0),  # Wave height in meters
            'direction': np.random.choice(directions),
            'breaking_type': np.random.choice(breaking_types)
        }
        
        # Save files
        image_file = output_path / f"dummy_image_{i:04d}.npy"
        label_file = output_path / f"dummy_image_{i:04d}_labels.npy"
        
        np.save(image_file, rgb_image)
        np.save(label_file, labels)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    
    print(f"\n✓ Successfully generated {num_samples} training samples")
    print(f"  Image files: {num_samples}")
    print(f"  Label files: {num_samples}")
    print(f"\nFiles saved to: {output_path}")
    print("\nYou can now run training with:")
    print(f"  python scripts/train.py --data-dir {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate dummy training data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for dummy data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    
    args = parser.parse_args()
    generate_dummy_data(args.output_dir, args.num_samples)

if __name__ == "__main__":
    main()
