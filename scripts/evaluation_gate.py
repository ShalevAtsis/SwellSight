#!/usr/bin/env python3
"""Fail if model metrics do not meet configured thresholds (CI gate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.utils.config import load_yaml_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwellSight evaluation quality gate")
    parser.add_argument(
        "--metrics",
        default="checkpoints/metrics.json",
        help="JSON file from training (trainer writes on best epoch)",
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="YAML with evaluation.thresholds section",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return 1

    with open(metrics_path, encoding="utf-8") as handle:
        metrics = json.load(handle)

    cfg = load_yaml_dict(args.config)
    thresholds = cfg.get("evaluation", {}).get("thresholds", {})
    max_height_mse = thresholds.get("max_height_mse", 1.0)
    min_direction_acc = thresholds.get("min_direction_acc", 0.0)
    min_breaking_acc = thresholds.get("min_breaking_acc", 0.0)

    ok = True
    height_mse = metrics.get("height_mse", metrics.get("height_loss", 999))
    if height_mse > max_height_mse:
        print(f"FAIL height_mse {height_mse:.4f} > {max_height_mse}")
        ok = False
    else:
        print(f"OK height_mse {height_mse:.4f}")

    dir_acc = metrics.get("direction_acc", 0)
    if dir_acc < min_direction_acc:
        print(f"FAIL direction_acc {dir_acc:.2%} < {min_direction_acc:.2%}")
        ok = False
    else:
        print(f"OK direction_acc {dir_acc:.2%}")

    brk_acc = metrics.get("breaking_acc", 0)
    if brk_acc < min_breaking_acc:
        print(f"FAIL breaking_acc {brk_acc:.2%} < {min_breaking_acc:.2%}")
        ok = False
    else:
        print(f"OK breaking_acc {brk_acc:.2%}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
