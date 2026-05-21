"""
Configuration management system for model parameters and settings.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"


@dataclass
class ModelConfig:
    """Configuration for model parameters."""

    depth_model_size: str = "large"
    depth_precision: str = "fp16"
    backbone_model: str = "dinov2-base"
    freeze_backbone: bool = True
    input_resolution: Tuple[int, int] = (518, 518)
    num_classes_direction: int = 3
    num_classes_breaking: int = 3


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = True
    save_checkpoint_every: int = 10
    validate_every: int = 5
    early_stopping_patience: int = 20
    height_loss_weight: float = 1.0
    direction_loss_weight: float = 1.0
    breaking_loss_weight: float = 1.0
    adaptive_loss_weighting: bool = True
    scheduler_type: str = "cosine"
    scheduler_step_on_batch: bool = False
    warmup_epochs: int = 5
    cosine_min_lr: float = 1e-6
    step_size: int = 30
    step_gamma: float = 0.1
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    log_interval: int = 100
    pretrain_epochs: int = 50
    finetune_epochs: int = 50
    synthetic_data_ratio: float = 0.8
    real_data_ratio: float = 0.2
    domain_adaptation_weight: float = 0.1


@dataclass
class DataConfig:
    """Configuration for data processing."""

    min_resolution: Tuple[int, int] = (640, 480)
    max_resolution: Tuple[int, int] = (3840, 2160)
    quality_threshold: float = 0.5
    augmentation_enabled: bool = True
    synthetic_data_ratio: float = 0.7
    target_resolution: Tuple[int, int] = (518, 518)
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1


@dataclass
class SystemConfig:
    """Configuration for system settings."""

    use_gpu: bool = True
    max_processing_time: float = 30.0
    confidence_threshold: float = 0.7
    log_level: str = "INFO"
    output_dir: str = "outputs"
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class PathsConfig:
    """Filesystem paths used by training and inference."""

    data_dir: str = "data"
    models_dir: str = "checkpoints"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    evaluation_dir: str = "evaluation"


@dataclass
class SwellSightConfig:
    """Complete SwellSight configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    extra: Dict[str, Any] = field(default_factory=dict)


def _to_tuple(value: Any, length: int = 2) -> Tuple[int, ...]:
    if value is None:
        return (518, 518) if length == 2 else tuple()
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value), int(value))


def _coerce_value(field_name: str, value: Any, field_type: Any) -> Any:
    if value is None:
        return value
    origin = getattr(field_type, "__origin__", None)
    if field_name.endswith("_resolution") or field_name == "min_resolution" or field_name == "max_resolution":
        return _to_tuple(value)
    if origin is tuple or field_type is tuple:
        return _to_tuple(value)
    if field_type is bool and isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if field_type is float and isinstance(value, str):
        return float(value)
    if field_type is int and isinstance(value, str):
        return int(value)
    return value


def _build_dataclass(cls, data: Optional[Dict[str, Any]]) -> Any:
    if not data:
        return cls()
    valid = {f.name: f.type for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    aliases = {
        "scheduler": "scheduler_type",
        "adaptive_weighting": "adaptive_loss_weighting",
        "height_loss_weight": "height_loss_weight",
        "direction_loss_weight": "direction_loss_weight",
        "breaking_loss_weight": "breaking_loss_weight",
    }
    for key, value in data.items():
        target = aliases.get(key, key)
        if target not in valid:
            continue
        kwargs[target] = _coerce_value(target, value, valid[target])
    return cls(**kwargs)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_dict(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file, resolving ``_base_`` inheritance."""
    path = Path(config_path)
    if not path.is_absolute():
        candidate = CONFIGS_DIR / path.name
        if candidate.exists():
            path = candidate
        elif (REPO_ROOT / path).exists():
            path = REPO_ROOT / path

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    base_key = data.get("_base_")
    if base_key:
        base_path = path.parent / base_key
        if not base_path.exists():
            base_path = CONFIGS_DIR / Path(base_key).name
        base_dict = load_yaml_dict(base_path)
        data = _merge_dicts(base_dict, data)

    return data


def dict_to_config(config_dict: Optional[Dict[str, Any]]) -> SwellSightConfig:
    """Convert a nested dictionary to ``SwellSightConfig``."""
    config_dict = config_dict or {}
    known = {"model", "training", "data", "system", "paths"}
    extra = {k: v for k, v in config_dict.items() if k not in known and k != "_base_"}
    return SwellSightConfig(
        model=_build_dataclass(ModelConfig, config_dict.get("model")),
        training=_build_dataclass(TrainingConfig, config_dict.get("training")),
        data=_build_dataclass(DataConfig, config_dict.get("data")),
        system=_build_dataclass(SystemConfig, config_dict.get("system")),
        paths=_build_dataclass(PathsConfig, config_dict.get("paths")),
        extra=extra,
    )


def config_to_dict(config: SwellSightConfig) -> Dict[str, Any]:
    """Convert ``SwellSightConfig`` to a plain dictionary."""
    result = {
        "model": asdict(config.model),
        "training": asdict(config.training),
        "data": asdict(config.data),
        "system": asdict(config.system),
        "paths": asdict(config.paths),
    }
    if config.extra:
        result.update(config.extra)
    return result


class ConfigManager:
    """Configuration manager for SwellSight system."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: SwellSightConfig = SwellSightConfig()
        if self.config_path and self.config_path.exists():
            self.load_config()

    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> SwellSightConfig:
        if config_path:
            self.config_path = Path(config_path)

        if not self.config_path or not self.config_path.exists():
            logger.warning("Config file not found: %s. Using defaults.", self.config_path)
            self.config = SwellSightConfig()
            return self.config

        try:
            config_dict = load_yaml_dict(self.config_path)
            self.config = dict_to_config(config_dict)
            logger.info("Configuration loaded from %s", self.config_path)
        except Exception as exc:
            logger.error("Failed to load config from %s: %s", self.config_path, exc)
            self.config = SwellSightConfig()

        return self.config

    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        if config_path:
            self.config_path = Path(config_path)
        if not self.config_path:
            raise ValueError("No config path specified")

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = config_to_dict(self.config)
        with open(self.config_path, "w", encoding="utf-8") as handle:
            if self.config_path.suffix.lower() in (".yaml", ".yml"):
                yaml.dump(payload, handle, default_flow_style=False, indent=2)
            else:
                json.dump(payload, handle, indent=2)

    def validate_config(self) -> bool:
        if not self.config:
            return False
        try:
            assert self.config.model.depth_model_size in ("small", "base", "large")
            assert self.config.model.depth_precision in ("fp16", "fp32")
            assert len(self.config.model.input_resolution) == 2
            assert self.config.training.batch_size > 0
            assert self.config.training.learning_rate > 0
            assert self.config.training.num_epochs > 0
            assert 0.0 <= self.config.data.quality_threshold <= 1.0
            assert self.config.system.max_processing_time > 0
            assert 0.0 <= self.config.system.confidence_threshold <= 1.0
            return True
        except AssertionError as exc:
            logger.error("Configuration validation failed: %s", exc)
            return False

    def get_config(self) -> SwellSightConfig:
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        merged = _merge_dicts(config_to_dict(self.config), updates)
        self.config = dict_to_config(merged)

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SwellSightConfig:
        return dict_to_config(config_dict)

    def _config_to_dict(self, config: SwellSightConfig) -> Dict[str, Any]:
        return config_to_dict(config)


def load_config(config_path: Union[str, Path]) -> SwellSightConfig:
    return ConfigManager(config_path).get_config()
