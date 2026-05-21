#!/usr/bin/env python3
"""Build datasets/manifest.json from .npy training files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.data.manifest import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory with .npy samples")
    parser.add_argument("--output", default="data/manifests/dataset_manifest.json")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    manifest = build_manifest(args.data_dir, args.output, args.train_ratio)
    print(f"Wrote {len(manifest.entries)} entries to {args.output}")
    print(f"Stats: {manifest.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
