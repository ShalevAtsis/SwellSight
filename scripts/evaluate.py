#!/usr/bin/env python3
"""Evaluate a trained SwellSight checkpoint on labeled data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.data.datasets import WaveDataset
from swellsight.models.wave_model import WaveAnalysisModel, load_checkpoint_into_model
from swellsight.utils.config import ConfigManager
from swellsight.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SwellSight wave model")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(log_level="DEBUG" if args.debug else "INFO")
    logger = logging.getLogger("swellsight.evaluate")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ConfigManager(args.config).get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WaveAnalysisModel(config).to(device)
    load_checkpoint_into_model(model, args.model_path, device=device)
    model.eval()

    target_resolution = tuple(config.data.target_resolution)
    dataset = WaveDataset(
        data_dir=args.test_data,
        split="validation",
        train_ratio=0.8,
        target_resolution=target_resolution,
    )
    if len(dataset) == 0:
        logger.error("No evaluation samples in %s", args.test_data)
        return 1

    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    height_loss = torch.nn.MSELoss()
    direction_loss = torch.nn.CrossEntropyLoss()
    breaking_loss = torch.nn.CrossEntropyLoss()

    totals = {"loss": 0.0, "height_mse": 0.0, "direction_acc": 0.0, "breaking_acc": 0.0}
    count = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            labels = batch["labels"]
            h_target = labels["height"].to(device).view(-1, 1)
            d_target = labels["direction"].to(device)
            b_target = labels["breaking_type"].to(device)

            outputs = model.forward_training(inputs)
            loss_h = height_loss(outputs["height"], h_target)
            loss_d = direction_loss(outputs["direction"], d_target)
            loss_b = breaking_loss(outputs["breaking_type"], b_target)
            loss = loss_h + loss_d + loss_b

            batch_size = inputs.size(0)
            count += batch_size
            totals["loss"] += loss.item() * batch_size
            totals["height_mse"] += loss_h.item() * batch_size
            totals["direction_acc"] += (
                (outputs["direction"].argmax(1) == d_target).float().sum().item()
            )
            totals["breaking_acc"] += (
                (outputs["breaking_type"].argmax(1) == b_target).float().sum().item()
            )

    metrics = {
        "samples": count,
        "avg_loss": totals["loss"] / count,
        "height_mse": totals["height_mse"] / count,
        "direction_accuracy": totals["direction_acc"] / count,
        "breaking_accuracy": totals["breaking_acc"] / count,
        "checkpoint": str(Path(args.model_path).resolve()),
        "test_data": str(Path(args.test_data).resolve()),
    }

    report_path = output_dir / "metrics.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
