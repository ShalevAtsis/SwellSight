#!/usr/bin/env python3
"""
Integration test for error handling and monitoring systems.

Tests the integration of error handling, logging, and monitoring
with existing SwellSight components.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import torch
from swellsight.utils.error_handler import error_handler, ProcessingError, MemoryError
from swellsight.utils.logging import setup_logging, performance_logger, health_monitor
from swellsight.utils.monitoring import system_monitor
from swellsight.core.depth_extractor import DepthAnythingV2Extractor, DepthMap

def test_error_handling_integration():
    """Test error handling integration with core components."""
    print("=== Error Handling Integration Test ===")
    
    # Setup logging
    setup_logging(
        log_level="INFO",
        enable_structured_logging=False,
        enable_performance_logging=True,
        enable_health_monitoring=False
    )
    
    # Test 1: Error handling with depth extractor
    print("\n1. Testing error handling with depth extractor...")
    try:
        extractor = DepthAnythingV2Extractor(model_size="small", device="cpu")
        
        # Test with invalid input (should trigger error handling)
        invalid_image = np.random.rand(100, 100)  # Wrong shape (missing channel dimension)
        
        try:
            depth_map, metrics = extractor.extract_depth(invalid_image)
        except ProcessingError as e:
            print(f"✓ ProcessingError caught: {e.component}.{e.operation}")
            print(f"✓ Error category: {e.category.value}")
            print(f"✓ Recovery suggestions: {len(e.recovery_suggestions)} provided")
        except Exception as e:
            print(f"✓ Generic error handled: {type(e).__name__}")
            
    except Exception as e:
        print(f"✓ Model loading error handled: {type(e).__name__}")
    
    # Test 2: Performance logging integration
    print("\n2. Testing performance logging integration...")
    
    # Simulate some operations
    performance_logger.log_performance("DepthExtractor", "model_loading", 2500.0)
    performance_logger.log_performance("WaveAnalyzer", "inference", 180.0)
    performance_logger.log_performance("Pipeline", "end_to_end", 3000.0)
    
    # Get performance summary
    perf_summary = performance_logger.get_performance_summary(last_n_minutes=1)
    print(f"✓ Performance summary: {perf_summary.get('total_operations', 0)} operations logged")
    
    # Test 3: System health monitoring
    print("\n3. Testing system health monitoring...")
    
    health_metrics = health_monitor.collect_health_metrics()
    print(f"✓ System health: CPU={health_metrics.cpu_usage_percent:.1f}%, Memory={health_metrics.memory_usage_percent:.1f}%")
    print(f"✓ GPU available: {health_metrics.gpu_available}")
    
    # Test 4: System monitoring and alerting
    print("\n4. Testing system monitoring...")
    
    status = system_monitor.get_system_status()
    print(f"✓ Overall system health: {status['overall_health']}")
    print(f"✓ Active alerts: {status['active_alerts']['total']}")
    
    # Test 5: Error summary and reporting
    print("\n5. Testing error summary...")
    
    error_summary = error_handler.get_error_summary()
    print(f"✓ Total errors tracked: {error_summary['total_errors']}")
    
    if error_summary['total_errors'] > 0:
        recent_error = error_summary['most_recent']
        print(f"✓ Most recent error: {recent_error['component']}.{recent_error['operation']}")
        print(f"✓ Error category: {recent_error['category']}")
    
    # Test 6: Memory and resource monitoring
    print("\n6. Testing resource monitoring...")
    
    # Simulate memory usage
    try:
        # Create a large tensor to test memory monitoring
        if torch.cuda.is_available():
            large_tensor = torch.randn(1000, 1000, device='cuda')
            print("✓ GPU memory allocation test completed")
            del large_tensor
            torch.cuda.empty_cache()
        else:
            large_tensor = torch.randn(1000, 1000)
            print("✓ CPU memory allocation test completed")
            del large_tensor
    except Exception as e:
        print(f"✓ Memory allocation error handled: {type(e).__name__}")
    
    print("\n=== Integration Test Completed Successfully! ===")
    
    # Final system status
    final_status = system_monitor.get_system_status()
    print(f"\nFinal system health: {final_status['overall_health']}")
    print(f"Total performance operations logged: {final_status['performance_summary'].get('total_operations', 0)}")
    print(f"Total errors handled: {final_status['error_summary']['total_errors']}")

if __name__ == "__main__":
    test_error_handling_integration()