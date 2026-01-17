#!/usr/bin/env python3
"""
Basic test for wave height regression head functionality.
Tests core functionality without property-based testing complexity.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from swellsight.core.wave_analyzer import DINOv2WaveAnalyzer
    from swellsight.core.depth_extractor import DepthMap
    SWELLSIGHT_AVAILABLE = True
except ImportError as e:
    print(f"SwellSight modules not available: {e}")
    SWELLSIGHT_AVAILABLE = False


def test_basic_wave_height_functionality():
    """Test basic wave height prediction functionality."""
    if not SWELLSIGHT_AVAILABLE:
        print("❌ SwellSight modules not available")
        return False
    
    print("Testing basic wave height functionality...")
    
    try:
        # Create analyzer
        analyzer = DINOv2WaveAnalyzer(backbone_model="dinov2_vitb14", freeze_backbone=True)
        analyzer.eval()
        
        # Set deterministic behavior
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test data
        height, width = 128, 128
        rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 255
        depth_array = np.random.rand(height, width).astype(np.float32)
        
        depth_map = DepthMap(
            data=depth_array,
            resolution=(height, width),
            quality_score=0.8,
            edge_preservation=0.7
        )
        
        # Test wave analysis
        wave_metrics = analyzer.analyze_waves(rgb_array, depth_map)
        
        # Basic validation
        print(f"  Height (meters): {wave_metrics.height_meters:.2f}")
        print(f"  Height (feet): {wave_metrics.height_feet:.2f}")
        print(f"  Height confidence: {wave_metrics.height_confidence:.3f}")
        print(f"  Direction: {wave_metrics.direction}")
        print(f"  Direction confidence: {wave_metrics.direction_confidence:.3f}")
        print(f"  Breaking type: {wave_metrics.breaking_type}")
        print(f"  Breaking confidence: {wave_metrics.breaking_confidence:.3f}")
        print(f"  Extreme conditions: {wave_metrics.extreme_conditions}")
        
        # Validate basic requirements
        assert 0.1 <= wave_metrics.height_meters <= 15.0, f"Height {wave_metrics.height_meters}m outside reasonable range"
        assert wave_metrics.height_meters > 0, "Height must be positive"
        assert 0.0 <= wave_metrics.height_confidence <= 1.0, f"Height confidence {wave_metrics.height_confidence} outside [0,1]"
        
        # Test unit conversion
        expected_feet = wave_metrics.height_meters * 3.28084
        conversion_error = abs(wave_metrics.height_feet - expected_feet)
        assert conversion_error < 0.001, f"Unit conversion error: {conversion_error}"
        
        # Test extreme condition detection
        is_extreme_height = wave_metrics.height_meters < 0.5 or wave_metrics.height_meters > 8.0
        if is_extreme_height:
            assert wave_metrics.extreme_conditions, "Extreme height should be flagged"
        
        print("✅ Basic wave height functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deterministic_behavior():
    """Test that the model produces consistent results for the same input."""
    if not SWELLSIGHT_AVAILABLE:
        print("❌ SwellSight modules not available")
        return False
    
    print("Testing deterministic behavior...")
    
    try:
        # Create analyzer
        analyzer = DINOv2WaveAnalyzer(backbone_model="dinov2_vitb14", freeze_backbone=True)
        analyzer.eval()
        
        # Create test data
        height, width = 64, 64  # Smaller for faster testing
        rgb_array = np.random.rand(height, width, 3).astype(np.float32) * 255
        depth_array = np.random.rand(height, width).astype(np.float32)
        
        depth_map = DepthMap(
            data=depth_array,
            resolution=(height, width),
            quality_score=0.8,
            edge_preservation=0.7
        )
        
        # Run multiple predictions with same input
        results = []
        for i in range(3):
            torch.manual_seed(42)  # Set seed before each prediction
            np.random.seed(42)
            
            with torch.no_grad():
                wave_metrics = analyzer.analyze_waves(rgb_array, depth_map)
                results.append({
                    'height': wave_metrics.height_meters,
                    'height_confidence': wave_metrics.height_confidence,
                    'direction': wave_metrics.direction,
                    'breaking_type': wave_metrics.breaking_type
                })
        
        # Check consistency
        for i in range(1, len(results)):
            height_diff = abs(results[i]['height'] - results[0]['height'])
            conf_diff = abs(results[i]['height_confidence'] - results[0]['height_confidence'])
            
            print(f"  Run {i+1}: Height={results[i]['height']:.3f}, Confidence={results[i]['height_confidence']:.3f}")
            
            # Allow small numerical differences due to floating point precision
            assert height_diff < 0.001, f"Height inconsistency: {height_diff}"
            assert conf_diff < 0.001, f"Confidence inconsistency: {conf_diff}"
            assert results[i]['direction'] == results[0]['direction'], "Direction inconsistency"
            assert results[i]['breaking_type'] == results[0]['breaking_type'], "Breaking type inconsistency"
        
        print("✅ Deterministic behavior test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Deterministic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running Basic Wave Height Tests...")
    print("=" * 50)
    
    success1 = test_basic_wave_height_functionality()
    success2 = test_deterministic_behavior()
    
    if success1 and success2:
        print("\n🎉 All basic tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)