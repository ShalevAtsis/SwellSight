"""
Integration test for wave_model.py, trainer.py, and train.py
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from swellsight.models.wave_model import WaveAnalysisModel
from swellsight.training.trainer import WaveAnalysisTrainer
from swellsight.utils.config import ConfigManager

def test_model_initialization():
    """Test that model can be initialized with config."""
    print("\n=== Testing Model Initialization ===")
    
    # Test with dict config
    config_dict = {
        'model': {
            'backbone': 'dinov2_vitb14',
            'freeze_backbone': True,
            'input_channels': 4,
            'num_classes_direction': 3,
            'num_classes_breaking': 3
        }
    }
    
    model = WaveAnalysisModel(config_dict)
    print("✓ Model initialized with dict config")
    
    # Test with ConfigManager
    config_manager = ConfigManager("configs/training.yaml")
    config = config_manager.get_config()
    
    model2 = WaveAnalysisModel(config)
    print("✓ Model initialized with ConfigManager")
    
    return True

def test_model_forward():
    """Test that model forward pass works."""
    print("\n=== Testing Model Forward Pass ===")
    
    config_dict = {
        'model': {
            'backbone': 'dinov2_vitb14',
            'freeze_backbone': True,
            'input_channels': 4,
            'num_classes_direction': 3,
            'num_classes_breaking': 3
        }
    }
    
    model = WaveAnalysisModel(config_dict)
    model.eval()
    
    # Create dummy input (batch_size=2, channels=4, height=224, width=224)
    dummy_input = torch.randn(2, 4, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Check outputs
    assert 'height' in outputs, "Missing 'height' output"
    assert 'direction' in outputs, "Missing 'direction' output"
    assert 'breaking_type' in outputs, "Missing 'breaking_type' output"
    
    assert outputs['height'].shape == (2, 1), f"Wrong height shape: {outputs['height'].shape}"
    assert outputs['direction'].shape == (2, 3), f"Wrong direction shape: {outputs['direction'].shape}"
    assert outputs['breaking_type'].shape == (2, 3), f"Wrong breaking_type shape: {outputs['breaking_type'].shape}"
    
    print(f"✓ Forward pass successful")
    print(f"  Height output shape: {outputs['height'].shape}")
    print(f"  Direction output shape: {outputs['direction'].shape}")
    print(f"  Breaking type output shape: {outputs['breaking_type'].shape}")
    
    return True

def test_trainer_initialization():
    """Test that trainer can be initialized."""
    print("\n=== Testing Trainer Initialization ===")
    
    # Test with dict config
    config_dict = {
        'model': {
            'backbone': 'dinov2_vitb14',
            'freeze_backbone': True,
            'input_channels': 4,
            'num_classes_direction': 3,
            'num_classes_breaking': 3
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 10,
            'weight_decay': 0.01,
            'optimizer': 'AdamW',
            'save_checkpoint_every': 5,
            'loss_weights': {
                'height': 1.0,
                'direction': 1.0,
                'breaking_type': 1.0
            }
        },
        'logging': {
            'save_dir': 'test_checkpoints'
        }
    }
    
    trainer = WaveAnalysisTrainer(config_dict)
    print("✓ Trainer initialized with dict config")
    
    # Test with ConfigManager
    config_manager = ConfigManager("configs/training.yaml")
    config = config_manager.get_config()
    
    trainer2 = WaveAnalysisTrainer(config)
    print("✓ Trainer initialized with ConfigManager")
    
    return True

def test_training_loop():
    """Test that training loop runs without errors."""
    print("\n=== Testing Training Loop ===")
    
    config_dict = {
        'model': {
            'backbone': 'dinov2_vitb14',
            'freeze_backbone': True,
            'input_channels': 4,
            'num_classes_direction': 3,
            'num_classes_breaking': 3
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'weight_decay': 0.01,
            'optimizer': 'AdamW',
            'save_checkpoint_every': 1,
            'loss_weights': {
                'height': 1.0,
                'direction': 1.0,
                'breaking_type': 1.0
            }
        },
        'logging': {
            'save_dir': 'test_checkpoints'
        }
    }
    
    trainer = WaveAnalysisTrainer(config_dict)
    
    # Create dummy data
    num_samples = 16
    inputs = torch.randn(num_samples, 4, 224, 224)
    heights = torch.randn(num_samples, 1)
    directions = torch.randint(0, 3, (num_samples,))
    breaking_types = torch.randint(0, 3, (num_samples,))
    
    # Create dataset and loaders
    train_data = []
    for i in range(num_samples):
        train_data.append({
            'input': inputs[i],
            'labels': {
                'height': heights[i],
                'direction': directions[i],
                'breaking_type': breaking_types[i]
            }
        })
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(train_data[:8], batch_size=4, shuffle=False)
    
    # Run training for 2 epochs
    print("Running training for 2 epochs...")
    trainer.train(train_loader, val_loader, num_epochs=2)
    
    print("✓ Training loop completed successfully")
    
    # Check that checkpoint was saved
    checkpoint_path = Path('test_checkpoints') / 'best_model.pth'
    assert checkpoint_path.exists(), "Best model checkpoint not found"
    print(f"✓ Checkpoint saved at {checkpoint_path}")
    
    return True

def test_checkpoint_loading():
    """Test that checkpoints can be loaded."""
    print("\n=== Testing Checkpoint Loading ===")
    
    config_dict = {
        'model': {
            'backbone': 'dinov2_vitb14',
            'freeze_backbone': True,
            'input_channels': 4,
            'num_classes_direction': 3,
            'num_classes_breaking': 3
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'weight_decay': 0.01,
            'optimizer': 'AdamW',
            'save_checkpoint_every': 1,
            'loss_weights': {
                'height': 1.0,
                'direction': 1.0,
                'breaking_type': 1.0
            }
        },
        'logging': {
            'save_dir': 'test_checkpoints'
        }
    }
    
    trainer = WaveAnalysisTrainer(config_dict)
    
    checkpoint_path = Path('test_checkpoints') / 'best_model.pth'
    if checkpoint_path.exists():
        epoch, metrics = trainer.load_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint loaded (Epoch {epoch+1})")
        print(f"  Metrics: {metrics}")
    else:
        print("⚠ No checkpoint found to load (run test_training_loop first)")
    
    return True

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Wave Model & Trainer Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Model Forward Pass", test_model_forward),
        ("Trainer Initialization", test_trainer_initialization),
        ("Training Loop", test_training_loop),
        ("Checkpoint Loading", test_checkpoint_loading),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
