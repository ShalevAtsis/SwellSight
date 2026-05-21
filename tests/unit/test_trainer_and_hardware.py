"""Unit tests for trainer utilities and hardware fallback (P1-T15)."""

from __future__ import annotations

import pytest
import torch

from swellsight.models.wave_model import WaveAnalysisModel
from swellsight.utils.config import SwellSightConfig
from swellsight.utils.hardware import HardwareManager


def test_wave_model_forward_cpu():
    config = SwellSightConfig()
    model = WaveAnalysisModel(config)
    model.eval()
    x = torch.randn(2, 4, 224, 224)
    with torch.no_grad():
        out = model.forward_training(x)
    assert out["height"].shape == (2, 1)
    assert out["direction"].shape[0] == 2
    assert out["breaking_type"].shape[0] == 2


def test_hardware_manager_device_selection():
    hw = HardwareManager()
    device = hw.get_device()
    assert device.type in ("cpu", "cuda")


def test_trainer_learning_rate_adjust():
    from swellsight.training.trainer import WaveAnalysisTrainer

    config = SwellSightConfig()
    trainer = WaveAnalysisTrainer(config, log_dir=None, callbacks=__import__(
        "swellsight.training.callbacks", fromlist=["TrainingCallbacks"]
    ).TrainingCallbacks())
    trainer._set_learning_rate(1e-5)
    assert trainer.learning_rate == 1e-5
