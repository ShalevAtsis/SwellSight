#!/usr/bin/env python3
"""Validate or create SwellSight standard data directories."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.data.layout import layout_help, validate_layout


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SwellSight data directory layout")
    parser.add_argument("--create-missing", action="store_true", help="Create missing standard dirs")
    args = parser.parse_args()

    report = validate_layout(create_missing=args.create_missing)
    print(layout_help())
    print()
    if report.existing:
        print("Existing:")
        for d in report.existing:
            print(f"  [OK] {d}")
    if report.missing:
        print("Missing:")
        for d in report.missing:
            print(f"  [--] {d}")
        return 1
    print("\nLayout OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
