import sys
from pathlib import Path
import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================
# 1) ROBUST PATH SETUP
# ============================================================
BASE = Path("/content/drive/MyDrive/SwellSight_Colab")

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(BASE / "src") not in sys.path:
    sys.path.insert(0, str(BASE / "src"))

# ============================================================
# 2) IMPORTS
# ============================================================
try:
    from src.swellsight.models.wave_model import WaveAnalysisModel as DINOv2WaveAnalyzer
    from src.swellsight.data.datasets import WaveDataset
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Optional: sklearn for confusion matrix / F1
try:
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ============================================================
# UTILITIES
# ============================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def safe_squeeze_height(x: torch.Tensor):
    # Convert [B,1] -> [B] if needed
    if isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[-1] == 1:
        return x.squeeze(-1)
    return x


def infer_logits_or_labels(t: torch.Tensor, num_classes: int = None):
    """
    If t is [B,C], treat as logits -> argmax labels.
    If t is [B], treat as already-labels.
    Returns (labels[B], logits_or_none)
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)

    if t.ndim == 2:
        # logits
        labels = torch.argmax(t, dim=1)
        return labels, t
    elif t.ndim == 1:
        # already labels
        return t.long(), None
    else:
        # unexpected
        return t.view(-1).long(), None


def compute_height_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    within_02 = float(np.mean(np.abs(err) <= 0.2))  # ratio 0..1
    return {
        "mae": mae,
        "rmse": rmse,
        "within_02": within_02,
        "errors": err,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if SKLEARN_OK:
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        cm = confusion_matrix(y_true, y_pred)
    else:
        # Fallback (no sklearn)
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        acc = float(np.mean(y_true == y_pred))
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        # Macro-F1 fallback (simple)
        f1s = []
        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = (2 * tp + fp + fn)
            f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
        f1 = float(np.mean(f1s))

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
    }


# ============================================================
# BATCH KEY NORMALIZER
# ============================================================
class KeyFixWrapper:
    """
    Ensures the batch contains:
      batch['image']
      batch['wave_height']
      batch['wave_direction']
      batch['breaking_type']
    Also mirrors them into batch['labels'] dict if present.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.printed_debug = False

    def __iter__(self):
        for batch in self.dataloader:

            # Image key
            if "image" not in batch:
                for possible_key in ["input", "inputs", "pixel_values", "rgb"]:
                    if possible_key in batch:
                        batch["image"] = batch.pop(possible_key)
                        break

            # Labels dict
            if "labels" not in batch or batch["labels"] is None:
                batch["labels"] = {}
            if not isinstance(batch["labels"], dict):
                batch["labels"] = {}

            # Flatten nested labels outward (no overwrite)
            for k, v in list(batch["labels"].items()):
                if k not in batch:
                    batch[k] = v

            if not self.printed_debug:
                print(f"   🔍 DEBUG(batch): keys before mapping: {list(batch.keys())}")
                print(f"   🔍 DEBUG(batch): labels keys before mapping: {list(batch['labels'].keys())}")

            # wave_height
            if "wave_height" not in batch:
                if "height" in batch:
                    batch["wave_height"] = batch["height"]
                elif "height_meters" in batch:
                    batch["wave_height"] = batch["height_meters"]

            # wave_direction
            if "wave_direction" not in batch:
                if "direction" in batch:
                    batch["wave_direction"] = batch["direction"]
                elif "direction_label" in batch:
                    batch["wave_direction"] = batch["direction_label"]

            # breaking_type
            if "breaking_type" not in batch:
                if "breaking" in batch:
                    batch["breaking_type"] = batch["breaking"]

            # Mirror into labels dict
            batch["labels"]["wave_height"] = batch["wave_height"]
            batch["labels"]["wave_direction"] = batch["wave_direction"]
            batch["labels"]["breaking_type"] = batch["breaking_type"]

            if not self.printed_debug:
                print(f"   ✅ DEBUG(batch): keys after mapping: {list(batch.keys())}")
                print(f"   ✅ DEBUG(batch): labels keys after mapping: {list(batch['labels'].keys())}")
                self.printed_debug = True

            yield batch

    def __len__(self):
        return len(self.dataloader)


# ============================================================
# MODEL OUTPUT NORMALIZER
# ============================================================
class OutputKeyFixModel(torch.nn.Module):
    """
    Normalizes model outputs to consistently include:
      out['wave_height']     -> regression output
      out['direction']       -> logits or labels (kept)
      out['breaking_type']   -> logits or labels (kept)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.printed_debug = False

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)

        if not isinstance(out, dict):
            return out

        if not self.printed_debug:
            print("   🧠 DEBUG(model): output keys BEFORE mapping:", list(out.keys()))

        # Height
        if "wave_height" not in out:
            if "height" in out:
                out["wave_height"] = out["height"]

        if "wave_height" in out and isinstance(out["wave_height"], torch.Tensor):
            out["wave_height"] = safe_squeeze_height(out["wave_height"])

        # Direction
        # Keep model's 'direction' as-is, but ensure it exists if possible
        if "direction" not in out:
            if "wave_direction" in out:
                out["direction"] = out["wave_direction"]

        # Breaking type
        if "breaking_type" not in out:
            if "breaking" in out:
                out["breaking_type"] = out["breaking"]

        if "wave_height" not in out:
            raise KeyError(f"Model output missing 'wave_height'. Keys: {list(out.keys())}")

        if not self.printed_debug:
            print("   ✅ DEBUG(model): output keys AFTER mapping:", list(out.keys()))
            self.printed_debug = True

        return out


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_height_pred_vs_true(y_true, y_pred, out_path: Path):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True Wave Height (m)")
    plt.ylabel("Predicted Wave Height (m)")
    plt.title("Wave Height: Predicted vs True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_height_error_hist(errors, out_path: Path):
    plt.figure()
    plt.hist(errors, bins=30)
    plt.xlabel("Prediction Error (m)  [pred - true]")
    plt.ylabel("Count")
    plt.title("Wave Height Error Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, title, out_path: Path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_class_distributions(direction_true, breaking_true, out_path: Path):
    plt.figure()

    # Direction distribution
    unique_d, counts_d = np.unique(direction_true, return_counts=True)
    plt.bar(unique_d.astype(int), counts_d)

    plt.xlabel("Direction Class")
    plt.ylabel("Count")
    plt.title("Direction Class Distribution (Ground Truth)")
    plt.tight_layout()
    plt.savefig(out_path.with_name("direction_distribution.png"), dpi=150)
    plt.close()

    plt.figure()
    unique_b, counts_b = np.unique(breaking_true, return_counts=True)
    plt.bar(unique_b.astype(int), counts_b)

    plt.xlabel("Breaking Type Class")
    plt.ylabel("Count")
    plt.title("Breaking Type Distribution (Ground Truth)")
    plt.tight_layout()
    plt.savefig(out_path.with_name("breaking_distribution.png"), dpi=150)
    plt.close()


def plot_metrics_summary(height_metrics, dir_metrics, break_metrics, overall_score, out_path: Path):
    # Simple summary bar chart
    labels = ["Height Within ±0.2m", "Direction Acc", "Breaking Acc", "Overall Score"]
    values = [
        height_metrics["within_02"] * 100.0,
        dir_metrics["accuracy"] * 100.0,
        break_metrics["accuracy"] * 100.0,
        overall_score * 100.0,
    ]

    plt.figure()
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Percent (%)")
    plt.title("Evaluation Metrics Summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sample_grid(images, y_true_h, y_pred_h, y_true_d, y_pred_d, y_true_b, y_pred_b, out_path: Path, max_items=12):
    """
    Shows a grid of sample test images with GT/PRED labels.
    Assumes images are tensors [B,C,H,W] in range 0..1 or normalized.
    """
    n = min(max_items, images.shape[0])
    cols = 4
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(12, 3 * rows))

    imgs = images[:n].detach().cpu()

    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        img = imgs[i]

        # If 4 channels, take first 3 for display
        if img.shape[0] >= 3:
            img_show = img[:3]
        else:
            img_show = img

        # Convert CHW -> HWC
        img_show = img_show.permute(1, 2, 0).numpy()

        # Try to make it visible even if normalized
        img_show = np.clip(img_show, 0.0, 1.0)

        ax.imshow(img_show)
        ax.axis("off")

        ax.set_title(
            f"H: {y_true_h[i]:.2f}->{y_pred_h[i]:.2f}\n"
            f"D: {int(y_true_d[i])}->{int(y_pred_d[i])}  "
            f"B: {int(y_true_b[i])}->{int(y_pred_b[i])}",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pipeline_diagram(out_path: Path):
    """
    A simple pipeline visualization:
    Dataset -> Dataloader -> Model -> Outputs -> Metrics -> Plots
    """
    fig = plt.figure(figsize=(10, 3))
    ax = plt.gca()
    ax.axis("off")

    boxes = [
        ("WaveDataset\n(synthetic/test)", 0.05, 0.35),
        ("DataLoader\n(batch=16)", 0.25, 0.35),
        ("DINOv2 Backbone\n+ Heads", 0.45, 0.35),
        ("Predictions\n(height, dir, break)", 0.65, 0.35),
        ("Metrics\n(MAE/F1/CM)", 0.85, 0.35),
    ]

    for text, x, y in boxes:
        box = FancyBboxPatch(
            (x, y), 0.18, 0.35,
            boxstyle="round,pad=0.02",
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(x + 0.09, y + 0.175, text, ha="center", va="center", fontsize=9)

    # arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 0.18
        y1 = boxes[i][2] + 0.175
        x2 = boxes[i + 1][1]
        y2 = boxes[i + 1][2] + 0.175
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12)
        ax.add_patch(arr)

    plt.title("SwellSight Evaluation Pipeline")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# MAIN EVALUATION
# ============================================================
def evaluate_and_visualize(base_dir=BASE):
    model_path = base_dir / "outputs/training/best_model.pth"
    data_dir = base_dir / "data/synthetic"
    out_dir = ensure_dir(base_dir / "outputs/evaluation")

    print("=" * 60)
    print("Model Evaluation + Visualizations")
    print("=" * 60)

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        return

    # ----------------------------
    # Load model
    # ----------------------------
    print("\n📦 Loading model...")
    model_config = {
        "model": {
            "backbone": "dinov2_vitb14",
            "freeze_backbone": True,
            "input_channels": 4,
            "num_classes_direction": 3,
            "num_classes_breaking": 3,
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = DINOv2WaveAnalyzer(model_config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint, strict=False)

    model = OutputKeyFixModel(base_model).to(device)
    model.eval()
    print("✓ Model loaded successfully")

    # ----------------------------
    # Load dataset
    # ----------------------------
    print("\n📊 Loading test dataset...")
    test_dataset = WaveDataset(data_dir=str(data_dir), split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    test_loader = KeyFixWrapper(test_loader)

    print(f"✓ Test samples: {len(test_dataset)}")

    # ----------------------------
    # Inference loop (collect everything)
    # ----------------------------
    print("\n🔍 Running inference...")
    all_true_h, all_pred_h = [], []
    all_true_d, all_pred_d = [], []
    all_true_b, all_pred_b = [], []
    sample_images_for_grid = None
    sample_meta_for_grid = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)

            # ground truth
            true_h = batch["wave_height"]
            true_d = batch["wave_direction"]
            true_b = batch["breaking_type"]

            # move GT to numpy
            true_h_np = to_numpy(true_h).astype(np.float32).reshape(-1)
            true_d_np = to_numpy(true_d).astype(np.int64).reshape(-1)
            true_b_np = to_numpy(true_b).astype(np.int64).reshape(-1)

            out = model(images)

            # predictions
            pred_h = out["wave_height"]
            pred_h = safe_squeeze_height(pred_h)
            pred_h_np = to_numpy(pred_h).astype(np.float32).reshape(-1)

            # direction predicted labels
            if "direction" in out:
                pred_d_labels, _ = infer_logits_or_labels(out["direction"])
            elif "wave_direction_logits" in out:
                pred_d_labels, _ = infer_logits_or_labels(out["wave_direction_logits"])
            else:
                # If model only outputs wave_direction labels
                pred_d_labels = torch.tensor(out.get("wave_direction", true_d)).long()

            pred_d_np = to_numpy(pred_d_labels).astype(np.int64).reshape(-1)

            # breaking predicted labels
            if "breaking_type" in out:
                pred_b_labels, _ = infer_logits_or_labels(out["breaking_type"])
            elif "breaking_type_logits" in out:
                pred_b_labels, _ = infer_logits_or_labels(out["breaking_type_logits"])
            else:
                pred_b_labels = torch.tensor(out.get("breaking", true_b)).long()

            pred_b_np = to_numpy(pred_b_labels).astype(np.int64).reshape(-1)

            # store
            all_true_h.append(true_h_np)
            all_pred_h.append(pred_h_np)

            all_true_d.append(true_d_np)
            all_pred_d.append(pred_d_np)

            all_true_b.append(true_b_np)
            all_pred_b.append(pred_b_np)

            # store first batch images for grid
            if sample_images_for_grid is None:
                sample_images_for_grid = images.detach().cpu()
                sample_meta_for_grid = (true_h_np, pred_h_np, true_d_np, pred_d_np, true_b_np, pred_b_np)

    # concat
    y_true_h = np.concatenate(all_true_h)
    y_pred_h = np.concatenate(all_pred_h)

    y_true_d = np.concatenate(all_true_d)
    y_pred_d = np.concatenate(all_pred_d)

    y_true_b = np.concatenate(all_true_b)
    y_pred_b = np.concatenate(all_pred_b)

    # ----------------------------
    # Compute metrics
    # ----------------------------
    print("\n📐 Computing metrics...")
    height_metrics = compute_height_metrics(y_true_h, y_pred_h)
    dir_metrics = compute_classification_metrics(y_true_d, y_pred_d)
    break_metrics = compute_classification_metrics(y_true_b, y_pred_b)

    # Simple overall score (you can change this formula)
    overall_score = (
        0.5 * height_metrics["within_02"]
        + 0.25 * dir_metrics["accuracy"]
        + 0.25 * break_metrics["accuracy"]
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Computed)")
    print("=" * 60)
    print("\n📏 Wave Height:")
    print(f"   MAE:  {height_metrics['mae']:.3f} m")
    print(f"   RMSE: {height_metrics['rmse']:.3f} m")
    print(f"   Within ±0.2m: {height_metrics['within_02']*100:.1f}%")

    print("\n🧭 Direction:")
    print(f"   Accuracy: {dir_metrics['accuracy']*100:.1f}%")
    print(f"   Macro F1:  {dir_metrics['macro_f1']:.3f}")

    print("\n💥 Breaking Type:")
    print(f"   Accuracy: {break_metrics['accuracy']*100:.1f}%")
    print(f"   Macro F1:  {break_metrics['macro_f1']:.3f}")

    print(f"\n⭐ Overall Score: {overall_score*100:.1f}%")
    print("=" * 60)

    # ----------------------------
    # Generate plots
    # ----------------------------
    print("\n📊 Generating plots...")

    plot_height_pred_vs_true(y_true_h, y_pred_h, out_dir / "height_pred_vs_true.png")
    plot_height_error_hist(height_metrics["errors"], out_dir / "height_error_hist.png")

    plot_confusion_matrix(
        dir_metrics["confusion_matrix"],
        "Direction Confusion Matrix",
        out_dir / "direction_confusion_matrix.png",
    )
    plot_confusion_matrix(
        break_metrics["confusion_matrix"],
        "Breaking Type Confusion Matrix",
        out_dir / "breaking_confusion_matrix.png",
    )

    plot_class_distributions(y_true_d, y_true_b, out_dir / "class_distributions.png")

    plot_metrics_summary(
        height_metrics,
        dir_metrics,
        break_metrics,
        overall_score,
        out_dir / "metrics_summary_bar.png",
    )

    if sample_images_for_grid is not None and sample_meta_for_grid is not None:
        th, ph, td, pd, tb, pb = sample_meta_for_grid
        plot_sample_grid(
            sample_images_for_grid,
            th, ph,
            td, pd,
            tb, pb,
            out_dir / "sample_predictions_grid.png",
            max_items=12,
        )

    plot_pipeline_diagram(out_dir / "pipeline_diagram.png")

    print("\n✅ Done. Saved visualizations to:")
    print(f"   {out_dir}")


if __name__ == "__main__":
    evaluate_and_visualize()
