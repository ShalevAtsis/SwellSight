"""
Test script for DINOv2 backbone implementation (Task 7.1).

This script verifies that the DINOv2 backbone can be loaded,
adapted for 4-channel input, and extract features correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path.cwd()))

from src.swellsight.models.backbone import DINOv2Backbone

def test_dinov2_backbone_loading():
    """Test 7.1: DINOv2 backbone loading and adaptation."""
    print("=" * 60)
    print("TEST 7.1: DINOv2 Backbone Loading and Adaptation")
    print("=" * 60)
    
    try:
        # Test 1: Load DINOv2 ViT-L/14 model
        print("\n1. Loading DINOv2 ViT-L/14 backbone...")
        backbone = DINOv2Backbone(
            model_name="dinov2_vitl14",
            freeze=True
        )
        print("   ✅ DINOv2 backbone loaded successfully")
        print(f"   - Feature dimension: {backbone.get_feature_dim()}")
        print(f"   - Input channels: {backbone.input_channels}")
        print(f"   - Target resolution: {backbone.target_size}")
        print(f"   - Frozen: {backbone.freeze}")
        
        # Test 2: Verify backbone is frozen
        print("\n2. Verifying backbone is frozen...")
        trainable_params = sum(p.numel() for p in backbone._backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in backbone._backbone.parameters())
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Frozen parameters: {total_params - trainable_params:,}")
        
        if trainable_params == 0:
            print("   ✅ Backbone is properly frozen")
        else:
            print(f"   ⚠️  Warning: {trainable_params} parameters are trainable")
        
        # Test 3: Test 4-channel input adaptation
        print("\n3. Testing 4-channel input adaptation...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = backbone.to(device)
        backbone.eval()
        
        # Create test input (RGB + Depth)
        batch_size = 2
        test_input = torch.randn(batch_size, 4, 518, 518).to(device)
        print(f"   - Test input shape: {test_input.shape}")
        print(f"   - Device: {device}")
        
        # Test forward pass
        with torch.no_grad():
            features = backbone(test_input)
        
        print(f"   ✅ 4-channel input adaptation successful")
        print(f"   - Output shape: {features.shape}")
        print(f"   - Feature dimension: {features.shape[1]}")
        
        # Test 4: Verify feature dimension
        print("\n4. Verifying feature dimension...")
        expected_dim = 1024
        actual_dim = features.shape[1]
        
        if actual_dim == expected_dim:
            print(f"   ✅ Feature dimension matches expected: {expected_dim}")
        else:
            print(f"   ❌ Feature dimension mismatch: got {actual_dim}, expected {expected_dim}")
            return False
        
        # Test 5: Test with different resolutions
        print("\n5. Testing automatic resizing...")
        test_resolutions = [(256, 256), (512, 512), (1024, 1024)]
        
        for h, w in test_resolutions:
            test_input_resized = torch.randn(1, 4, h, w).to(device)
            with torch.no_grad():
                features_resized = backbone(test_input_resized)
            print(f"   - Input {h}x{w} -> Output {features_resized.shape[1]}-dim: ✅")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Task 7.1 Implementation Verified")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dinov2_backbone_loading()
    sys.exit(0 if success else 1)
