#!/usr/bin/env python3
"""Promote a training checkpoint into the model registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.mlops.registry import load_registry, promote_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. wave-v0.1.0")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--metrics", default="checkpoints/metrics.json")
    parser.add_argument("--manifest", default="data/manifests/dataset_manifest.json")
    parser.add_argument("--status", default="production", choices=["staging", "production", "archived"])
    parser.add_argument("--no-active", action="store_true")
    args = parser.parse_args()

    metrics = {}
    if Path(args.metrics).exists():
        with open(args.metrics, encoding="utf-8") as handle:
            metrics = json.load(handle)

    version = promote_checkpoint(
        version_id=args.version,
        checkpoint_path=args.checkpoint,
        metrics=metrics,
        data_manifest=args.manifest if Path(args.manifest).exists() else None,
        status=args.status,
        set_active=not args.no_active,
    )

    registry = load_registry()
    print(f"Promoted {version.id} -> {version.checkpoint}")
    print(f"Active version: {registry.active_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
