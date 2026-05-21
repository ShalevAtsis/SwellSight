import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from swellsight.models.losses import MultiTaskLoss
from swellsight.models.wave_model import WaveAnalysisModel
from swellsight.training.scheduler import create_lr_scheduler
from swellsight.training.callbacks import TrainingCallbacks, build_default_callbacks

logger = logging.getLogger(__name__)


class WaveAnalysisTrainer:
    """Trainer for the SwellSight wave analysis model."""

    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        log_dir: Optional[str] = None,
        callbacks: Optional[TrainingCallbacks] = None,
    ):
        self.config = config
        self._read_config(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = WaveAnalysisModel(config).to(self.device)

        self.criterion = MultiTaskLoss(
            height_weight=self.weights["height"],
            direction_weight=self.weights["direction"],
            breaking_weight=self.weights["breaking_type"],
            adaptive_weighting=self.adaptive_weighting,
        ).to(self.device)

        params = list(self.model.parameters()) + list(self.criterion.parameters())
        params = [p for p in params if p.requires_grad]

        if self.optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(
                params, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.Adam(params, lr=self.learning_rate)

        scheduler_type = getattr(self, "scheduler_type", "warmup_cosine")
        if scheduler_type == "cosine" and self.warmup_epochs > 0:
            scheduler_type = "warmup_cosine"

        self.scheduler = create_lr_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            num_epochs=self.num_epochs,
            warmup_epochs=self.warmup_epochs,
            min_lr=self.cosine_min_lr,
        )

        self.best_metrics: Dict[str, float] = {}
        self.pretrain_epochs = getattr(self, "pretrain_epochs", 50)
        self.finetune_epochs = getattr(self, "finetune_epochs", 20)
        self.finetune_learning_rate = getattr(self, "finetune_learning_rate", self.learning_rate * 0.1)

        self.callbacks = callbacks or build_default_callbacks(log_dir or str(self.save_dir.parent / "logs"))

        logger.info(
            "Trainer ready: device=%s trainable=%s lr=%s",
            self.device,
            f"{sum(p.numel() for p in params):,}",
            self.learning_rate,
        )

    def _read_config(self, config: Union[Dict[str, Any], Any]) -> None:
        if hasattr(config, "training"):
            train_conf = config.training
            log_conf = config.system
            self.batch_size = train_conf.batch_size
            self.learning_rate = train_conf.learning_rate
            self.num_epochs = train_conf.num_epochs
            self.weight_decay = train_conf.weight_decay
            self.gradient_clip_norm = train_conf.gradient_clip_norm
            self.optimizer_name = "AdamW"
            self.save_frequency = train_conf.save_checkpoint_every
            self.early_stopping_patience = train_conf.early_stopping_patience
            self.adaptive_weighting = train_conf.adaptive_loss_weighting
            self.scheduler_type = train_conf.scheduler_type
            self.warmup_epochs = train_conf.warmup_epochs
            self.cosine_min_lr = train_conf.cosine_min_lr
            self.weights = {
                "height": train_conf.height_loss_weight,
                "direction": train_conf.direction_loss_weight,
                "breaking_type": train_conf.breaking_loss_weight,
            }
            self.pretrain_epochs = train_conf.pretrain_epochs
            self.finetune_epochs = train_conf.finetune_epochs
            self.finetune_learning_rate = self.learning_rate * 0.1
            checkpoints = getattr(config, "paths", None)
            self.save_dir = (
                Path(checkpoints.checkpoints_dir)
                if checkpoints
                else Path(log_conf.output_dir) / "checkpoints"
            )
        else:
            train_conf = config.get("training", {})
            log_conf = config.get("logging", {})
            self.batch_size = train_conf.get("batch_size", 32)
            self.learning_rate = float(train_conf.get("learning_rate", 1e-4))
            self.num_epochs = train_conf.get("num_epochs", 100)
            self.weight_decay = float(train_conf.get("weight_decay", 0.01))
            self.gradient_clip_norm = float(train_conf.get("gradient_clip_norm", 1.0))
            self.optimizer_name = train_conf.get("optimizer", "AdamW")
            self.save_dir = Path(log_conf.get("save_dir", "checkpoints"))
            self.save_frequency = train_conf.get("save_checkpoint_every", 5)
            self.early_stopping_patience = train_conf.get("early_stopping_patience", 20)
            self.adaptive_weighting = train_conf.get("adaptive_weighting", True)
            self.scheduler_type = train_conf.get("scheduler", "warmup_cosine")
            self.warmup_epochs = train_conf.get("warmup_epochs", 5)
            self.cosine_min_lr = float(train_conf.get("min_lr", 1e-6))
            self.weights = train_conf.get(
                "loss_weights",
                {"height": 1.0, "direction": 1.0, "breaking_type": 1.0},
            )

    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        best_val_loss = float("inf")
        epochs_without_improve = 0

        logger.info("Starting training for %s epochs...", num_epochs)

        for epoch in range(num_epochs):
            logger.info("Epoch %s/%s", epoch + 1, num_epochs)
            self.callbacks.on_epoch_start(epoch, {"phase": "train"})

            train_metrics = self._run_epoch(train_loader, is_training=True)
            self._log_metrics(train_metrics, "Train")
            self.callbacks.on_epoch_end(epoch, {**train_metrics, "prefix": "train"})

            val_metrics = self._run_epoch(val_loader, is_training=False)
            self._log_metrics(val_metrics, "Val")
            self.callbacks.on_epoch_end(epoch, {**val_metrics, "prefix": "val"})

            if self.scheduler is not None and not isinstance(
                self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()

            current_val_loss = val_metrics["total_loss"]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_without_improve = 0
                self.best_metrics = val_metrics
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                self._write_metrics_json(val_metrics)
                logger.info("[BEST] validation loss %.4f", best_val_loss)
            else:
                epochs_without_improve += 1

            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            if epochs_without_improve >= self.early_stopping_patience:
                logger.info("Early stopping after %s epochs without improvement", epochs_without_improve)
                break

        logger.info("Training done. Best val loss: %.4f", best_val_loss)

    def train_sim_to_real(
        self,
        synthetic_train_loader,
        synthetic_val_loader,
        real_train_loader,
        real_val_loader,
        pretrain_epochs: Optional[int] = None,
        finetune_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Phase 1: synthetic pretrain. Phase 2: real finetune (lower LR)."""
        pretrain_epochs = pretrain_epochs or self.pretrain_epochs
        finetune_epochs = finetune_epochs or self.finetune_epochs

        logger.info("=== Sim-to-real Phase 1: synthetic pretrain (%s epochs) ===", pretrain_epochs)
        self.train(synthetic_train_loader, synthetic_val_loader, num_epochs=pretrain_epochs)
        pretrain_path = self.save_dir / "pretrain_best.pth"
        if (self.save_dir / "best_model.pth").exists():
            import shutil
            shutil.copy(self.save_dir / "best_model.pth", pretrain_path)

        logger.info("=== Sim-to-real Phase 2: real finetune (%s epochs) ===", finetune_epochs)
        self._set_learning_rate(self.finetune_learning_rate)
        self.train(real_train_loader, real_val_loader, num_epochs=finetune_epochs)

        return {
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": finetune_epochs,
            "pretrain_checkpoint": str(pretrain_path),
            "best_checkpoint": str(self.save_dir / "best_model.pth"),
            "best_metrics": self.best_metrics,
        }

    def _set_learning_rate(self, lr: float) -> None:
        self.learning_rate = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        logger.info("Learning rate set to %s", lr)

    def _compute_loss(
        self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if "height" in outputs:
            predictions = {
                "height_meters": outputs["height"],
                "direction_logits": outputs["direction"],
                "breaking_logits": outputs["breaking_type"],
            }
        else:
            predictions = outputs

        targets = {
            "height_meters": labels["height"].view(-1, 1),
            "direction_labels": labels["direction"],
            "breaking_labels": labels["breaking_type"],
        }
        return self.criterion(predictions, targets)

    def _run_epoch(self, loader, is_training: bool) -> Dict[str, float]:
        if is_training:
            self.model.train()
            self.criterion.train()
        else:
            self.model.eval()
            self.criterion.eval()

        total_loss = 0.0
        height_losses = []
        dir_accs = []
        break_accs = []

        pbar = tqdm(loader, desc="Training" if is_training else "Validating", leave=False)

        for batch in pbar:
            inputs = batch["input"].to(self.device)
            labels = batch["labels"]

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                if hasattr(self.model, "forward_training"):
                    outputs = self.model.forward_training(inputs)
                else:
                    raw = self.model(inputs)
                    outputs = {
                        "height": raw["height_meters"].view(-1, 1),
                        "direction": raw["direction_logits"],
                        "breaking_type": raw["breaking_logits"],
                    }

                loss_dict = self._compute_loss(outputs, labels)
                loss = loss_dict["total_loss"]

                if is_training:
                    loss.backward()
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_norm
                        )
                    self.optimizer.step()

            d_target = labels["direction"].to(self.device)
            b_target = labels["breaking_type"].to(self.device)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            height_losses.append(loss_dict["height_loss"].item())

            _, d_pred = torch.max(outputs["direction"], 1)
            dir_accs.append((d_pred == d_target).float().mean().item())
            _, b_pred = torch.max(outputs["breaking_type"], 1)
            break_accs.append((b_pred == b_target).float().mean().item())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        n = max(len(loader.dataset), 1)
        return {
            "total_loss": total_loss / n,
            "height_mse": float(np.mean(height_losses)),
            "direction_acc": float(np.mean(dir_accs)),
            "breaking_acc": float(np.mean(break_accs)),
        }

    def _log_metrics(self, metrics: Dict[str, float], prefix: str) -> None:
        logger.info(
            "  %s Loss: %.4f | Height MSE: %.4f | Dir Acc: %.2f%% | Brk Acc: %.2f%%",
            prefix,
            metrics["total_loss"],
            metrics["height_mse"],
            metrics["direction_acc"] * 100,
            metrics["breaking_acc"] * 100,
        )

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth"
        path = self.save_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "criterion_state_dict": self.criterion.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def _write_metrics_json(self, metrics: Dict[str, float]) -> None:
        path = self.save_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "criterion_state_dict" in checkpoint:
            self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
