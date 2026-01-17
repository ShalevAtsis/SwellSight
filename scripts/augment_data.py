import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --- SETUP PATHS ---
if 'BASE' not in globals():
    BASE = "/content/drive/MyDrive/SwellSight_Colab"
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.swellsight.data.augmentation import WaveAugmentation

def augment_dataset(input_dir, output_dir, augmentations_per_image=3):
    """Apply augmentations to synthetic dataset."""

    print("Initializing augmentation system...")
    augmenter = WaveAugmentation()
    print("✓ Augmenter initialized")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load synthetic images (filter out existing augs if re-running)
    image_files = list(input_path.glob('synthetic_*.npy'))
    image_files = [f for f in image_files if '_labels' not in f.name and '_depth' not in f.name]

    print(f"\nAugmenting {len(image_files)} images...")
    print(f"Creating {augmentations_per_image} variations per image")

    total_generated = 0
    
    # We use a distinct seed loop or rely on the random module's state. 
    # Since the class uses random.random(), no manual seed reset per image is strictly needed 
    # unless reproducibility is critical.

    for img_file in tqdm(image_files):
        # Load image and labels
        img = np.load(img_file)
        label_file = img_file.parent / f"{img_file.stem}_labels.npy"
        
        # Safety check for missing label files
        if not label_file.exists():
            continue
            
        labels = np.load(label_file, allow_pickle=True).item()

        # Generate augmentations
        for aug_idx in range(augmentations_per_image):
            
            # --- FIX: Use correct method and handle Dictionary return ---
            result = augmenter.augment_training_sample(img, preserve_labels=True)
            
            # Check if augmentation succeeded
            if not result.get('augmentation_success', False):
                continue

            # Extract the actual image array
            aug_img = result['augmented_image']

            # Save augmented image
            aug_file = output_path / f"{img_file.stem}_aug{aug_idx}.npy"
            np.save(aug_file, aug_img)

            # Copy labels 
            # (SAFE for this specific class because it only does weather/lighting, no geometry)
            aug_label_file = output_path / f"{img_file.stem}_aug{aug_idx}_labels.npy"
            np.save(aug_label_file, labels)

            total_generated += 1

    print(f"\n✓ Generated {total_generated} augmented images")
    print(f"✓ Total dataset size: {len(image_files) + total_generated} images")

if __name__ == "__main__":
    if 'DATA_DIR' not in globals():
        DATA_DIR = f"{BASE}/data"

    augment_dataset(
        f'{DATA_DIR}/synthetic',
        f'{DATA_DIR}/augmented',
        augmentations_per_image=3
    )