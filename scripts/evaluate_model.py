import sys
sys.path.insert(0, '.')

from pathlib import Path
import torch
from src.swellsight.evaluation.evaluator import ModelEvaluator
from src.swellsight.models.wave_analyzer import DINOv2WaveAnalyzer
from src.swellsight.data.datasets import WaveDataset
from torch.utils.data import DataLoader

def evaluate_model(model_path='models/checkpoints/best_model.pth'):
    """Evaluate the trained model."""
    
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    model = DINOv2WaveAnalyzer()
    model.load_state_dict(torch.load(model_path))
    print("✓ Model loaded")
    
    # Create test dataset
    print("\n📊 Loading test dataset...")
    test_dataset = WaveDataset(
        data_dir='data/synthetic',
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model)
    
    # Evaluate accuracy
    print("\n🔍 Evaluating accuracy...")
    accuracy_metrics = evaluator.evaluate_accuracy(test_loader)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\n📏 Wave Height:")
    print(f"   MAE: {accuracy_metrics.height_metrics.mae:.3f}m")
    print(f"   RMSE: {accuracy_metrics.height_metrics.rmse:.3f}m")
    print(f"   Within ±0.2m: {accuracy_metrics.height_metrics.accuracy_within_02m:.1%}")
    
    print(f"\n🧭 Direction:")
    print(f"   Accuracy: {accuracy_metrics.direction_metrics.accuracy:.1%}")
    print(f"   F1-Score: {accuracy_metrics.direction_metrics.macro_avg_f1:.3f}")
    
    print(f"\n💥 Breaking Type:")
    print(f"   Accuracy: {accuracy_metrics.breaking_type_metrics.accuracy:.1%}")
    print(f"   F1-Score: {accuracy_metrics.breaking_type_metrics.macro_avg_f1:.3f}")
    
    print(f"\n⭐ Overall Score: {accuracy_metrics.overall_score:.1%}")
    print("="*60)
    
    # Benchmark performance
    print("\n⚡ Benchmarking performance...")
    perf_metrics = evaluator.benchmark_performance(test_loader)
    
    print(f"\n   Inference Time: {perf_metrics['inference_time_ms']:.1f}ms")
    print(f"   Throughput: {perf_metrics['throughput_images_per_second']:.2f} images/sec")
    print(f"   Memory Usage: {perf_metrics['memory_usage_mb']:.1f}MB")

if __name__ == "__main__":
    evaluate_model()