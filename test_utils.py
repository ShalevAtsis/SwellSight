#!/usr/bin/env python3
"""
Test script for SwellSight Pipeline Utilities
Verifies that all utility functions work correctly
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_manager():
    """Test configuration management functionality"""
    print("Testing Configuration Manager...")
    
    try:
        from utils.config_manager import ConfigManager, load_config, validate_config
        
        # Test loading default config
        config = load_config("config.json")
        print(f"✓ Loaded configuration: {config['pipeline']['name']}")
        
        # Test validation
        is_valid = validate_config(config)
        print(f"✓ Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Test ConfigManager class
        manager = ConfigManager("config.json")
        loaded_config = manager.load_config()
        print(f"✓ ConfigManager loaded: {loaded_config['pipeline']['version']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration Manager test failed: {e}")
        return False

def test_data_validator():
    """Test data validation functionality"""
    print("\nTesting Data Validator...")
    
    try:
        from utils.data_validator import DataValidator, validate_image_quality
        
        # Create a test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        from PIL import Image
        
        # Save test image
        test_image_path = "test_image.jpg"
        Image.fromarray(test_image).save(test_image_path)
        
        # Test image validation
        result = validate_image_quality(test_image_path)
        print(f"✓ Image validation result: valid={result['valid']}, score={result['score']:.3f}")
        
        # Test depth map validation
        validator = DataValidator()
        test_depth = np.random.rand(256, 256).astype(np.float32)
        depth_result = validator.validate_depth_map_quality(test_depth)
        print(f"✓ Depth validation result: valid={depth_result['valid']}, score={depth_result['score']:.3f}")
        
        # Cleanup
        os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Data Validator test failed: {e}")
        return False

def test_memory_optimizer():
    """Test memory optimization functionality"""
    print("\nTesting Memory Optimizer...")
    
    try:
        from utils.memory_optimizer import MemoryOptimizer, get_optimal_batch_size, monitor_memory
        
        # Test batch size calculation
        batch_size = get_optimal_batch_size(max_batch_size=16)
        print(f"✓ Optimal batch size calculated: {batch_size}")
        
        # Test memory monitoring
        memory_info = monitor_memory()
        if memory_info:
            print(f"✓ Memory monitoring: {memory_info.get('system_percent', 0):.1f}% system usage")
        
        # Test MemoryOptimizer class
        optimizer = MemoryOptimizer()
        suggestions = optimizer.suggest_memory_optimizations()
        print(f"✓ Generated {len(suggestions)} optimization suggestions")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory Optimizer test failed: {e}")
        return False

def test_error_handler():
    """Test error handling functionality"""
    print("\nTesting Error Handler...")
    
    try:
        from utils.error_handler import ErrorHandler, retry_with_backoff
        
        # Test retry mechanism with a function that succeeds
        def test_function():
            return "success"
        
        result = retry_with_backoff(test_function, max_retries=2)
        print(f"✓ Retry mechanism test: {result}")
        
        # Test error handler class
        handler = ErrorHandler()
        recovery_instructions = handler.provide_recovery_instructions("memory_error", "test_stage")
        print(f"✓ Generated {len(recovery_instructions)} recovery instructions")
        
        return True
        
    except Exception as e:
        print(f"✗ Error Handler test failed: {e}")
        return False

def test_progress_tracker():
    """Test progress tracking functionality"""
    print("\nTesting Progress Tracker...")
    
    try:
        from utils.progress_tracker import ProgressTracker, create_progress_bar
        
        # Test progress bar creation
        progress_bar = create_progress_bar(10, "Testing")
        print("✓ Progress bar created successfully")
        
        # Test ProgressTracker class
        tracker = ProgressTracker()
        test_metrics = {
            'items_processed': 100,
            'duration_seconds': 30.5,
            'success_rate': 0.95
        }
        tracker.display_stage_summary("test_stage", test_metrics)
        print("✓ Stage summary displayed")
        
        return True
        
    except Exception as e:
        print(f"✗ Progress Tracker test failed: {e}")
        return False

def test_data_flow_manager():
    """Test data flow management functionality"""
    print("\nTesting Data Flow Manager...")
    
    try:
        from utils.data_flow_manager import DataFlowManager, save_stage_results, check_dependencies
        
        # Test saving stage results
        test_data = {
            'processed_items': 50,
            'quality_scores': [0.8, 0.9, 0.7],
            'output_paths': ['output1.jpg', 'output2.jpg']
        }
        
        success = save_stage_results(test_data, "test_stage", {"test": True})
        print(f"✓ Stage results saved: {success}")
        
        # Test dependency checking
        dep_status = check_dependencies("test_stage")
        print(f"✓ Dependency check completed: {dep_status['all_satisfied']}")
        
        # Test DataFlowManager class
        manager = DataFlowManager()
        pipeline_status = manager.get_pipeline_status()
        print(f"✓ Pipeline status retrieved: {len(pipeline_status.get('completed_stages', []))} stages completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Data Flow Manager test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("SwellSight Pipeline Utilities Test Suite")
    print("=" * 50)
    
    tests = [
        test_config_manager,
        test_data_validator,
        test_memory_optimizer,
        test_error_handler,
        test_progress_tracker,
        test_data_flow_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Utilities are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())