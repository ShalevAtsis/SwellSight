from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, target_size=(640, 480)):
    """Preprocess beach cam images."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    print(f"Preprocessing {len(images)} images...")

    for img_path in tqdm(images):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Resize to target size
        img_resized = cv2.resize(img, target_size)

        # Save preprocessed image
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), img_resized,
                   [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"✓ Preprocessed {len(images)} images")

# Call the function with the correct paths
preprocess_images(f"{DATA_DIR}/raw", f"{DATA_DIR}/processed")