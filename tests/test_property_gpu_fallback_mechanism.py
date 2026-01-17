#!/usr/bin/env python3
"""
Property-Based Test for GPU Fallback Mechanism
Feature: swellsight-pipeline-improvements, Property 18: GPU Fallback Mechanism
Validates: Requirements 4.3
"""

import sys
import pytest
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings, example, assume

# Add utils to path
sys.path.append('.')

from utils.error_handler import ErrorHandler, handle_gpu_memory_error

class TestGPUFallbackMechanism:
    """Property-based tests for GPU fallback mechanism"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.error_handler = ErrorHandler()
        self.log_messages = []
        self.mock_logger = Mock()
        self.mock_logger.warning = Mock(side_effect=lambda msg: self.log_messages.append(('warning', msg)))
        self.mock_logger.info = Mock(side_effect=lambda msg: self.log_messages.append(('info', msg)))
        self.mock_logger.error = Mock(side_effect=lambda msg: self.log_messages.append(('error', msg)))
    
    def test_basic_gpu_fallback_mechanism(self):
        """
        Basic test for GPU fallback mechanism
        
        Feature: swellsight-pipeline-improvements, Property 18: GPU Fallback Mechanism
        Validates: Requirements 4.3
        """
        operation_name = "test_operation"
        fallback_func = Mock(return_value="cpu_result")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.cuda.memory_allocated', return_value=1024*1024*1024), \
             patch('torch.cuda.memory_reserved', return_value=2*1024*1024*1024), \
             patch('utils.error_handler.logger', self.mock_logger):
            
            result = self.error_handler.handle_gpu_memory_error(operation_name, fallback_func)
            
            # Should return fallback result
            assert result == "cpu_result", "Should return fallback result"
            
            # Should call fallback function
            fallback_func.assert_called_once()
            
            # Should clear GPU cache
            mock_empty_cache.assert_called_once()
            
            # Should log appropriate messages
            log_messages = [msg for level, msg in self.log_messages]
            warning_logged = any("GPU memory error" in msg and operation_name in msg for msg in log_messages)
            assert warning_logged, f"Should log GPU memory warning for {operation_name}"
    
    @given(
        operation_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        has_fallback=st.booleans(),
        gpu_available=st.booleans()
    )
    @settings(max_examples=100, deadline=5000)
    @example(operation_name="depth_extraction", has_fallback=True, gpu_available=True)
    def test_gpu_fallback_property(self, operation_name: str, has_fallback: bool, gpu_available: bool):
        """
        Property: For any GPU operation failure, the system should fall back to 
        CPU processing with appropriate warnings
        
        Feature: swellsight-pipeline-improvements, Property 18: GPU Fallback Mechanism
        Validates: Requirements 4.3
        """
        # Ensure operation name is valid
        assume(len(operation_name.strip()) > 0)
        operation_name = operation_name.strip()
        
        # Mock fallback function
        fallback_result = f"cpu_result_for_{operation_name}"
        fallback_func = Mock(return_value=fallback_result) if has_fallback else None
        
        # Reset log messages
        self.log_messages.clear()
        
        # Mock GPU availability
        with patch('torch.cuda.is_available', return_value=gpu_available), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('torch.cuda.memory_allocated', return_value=1024*1024*1024), \
             patch('torch.cuda.memory_reserved', return_value=2*1024*1024*1024), \
             patch('utils.error_handler.logger', self.mock_logger):
            
            # Test GPU fallback mechanism
            result = self.error_handler.handle_gpu_memory_error(operation_name, fallback_func)
            
            # Property assertions
            if has_fallback:
                # Should return fallback result
                assert result == fallback_result, f"Should return fallback result for {operation_name}"
                # Should call fallback function
                fallback_func.assert_called_once()
            else:
                # Should return None when no fallback available
                assert result is None, f"Should return None when no fallback available for {operation_name}"
            
            # Should clear GPU cache if GPU is available
            if gpu_available:
                mock_empty_cache.assert_called_once()
            
            # Should log appropriate messages
            log_messages = [msg for level, msg in self.log_messages]
            
            if gpu_available:
                # Should log GPU memory warning
                warning_logged = any("GPU memory error" in msg and operation_name in msg for msg in log_messages)
                assert warning_logged, f"Should log GPU memory warning for {operation_name}"
            
            if has_fallback:
                # Should log fallback message
                fallback_logged = any("Falling back to CPU processing" in msg and operation_name in msg for msg in log_messages)
                assert fallback_logged, f"Should log CPU fallback message for {operation_name}"
            else:
                # Should log no fallback available
                no_fallback_logged = any("No fallback available" in msg and operation_name in msg for msg in log_messages)
                assert no_fallback_logged, f"Should log no fallback available for {operation_name}"
    
    def test_no_gpu_available_property(self):
        """
        Property: When GPU is not available, fallback should still work 
        but skip GPU-specific operations
        
        Feature: swellsight-pipeline-improvements, Property 18: GPU Fallback Mechanism
        Validates: Requirements 4.3
        """
        operation_name = "cpu_only_operation"
        fallback_func = Mock(return_value="cpu_result")
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('utils.error_handler.logger', self.mock_logger):
            
            result = self.error_handler.handle_gpu_memory_error(operation_name, fallback_func)
            
            # Should return fallback result
            assert result == "cpu_result", "Should return fallback result even without GPU"
            
            # Should call fallback function
            fallback_func.assert_called_once()
            
            # Should NOT call GPU cache clearing when GPU not available
            mock_empty_cache.assert_not_called()
            
            # Should still log fallback message
            fallback_logged = any("Falling back to CPU processing" in msg for level, msg in self.log_messages)
            assert fallback_logged, "Should log CPU fallback message even without GPU"
    
    def test_convenience_function_property(self):
        """
        Property: The convenience function should behave identically to 
        the class method
        
        Feature: swellsight-pipeline-improvements, Property 18: GPU Fallback Mechanism
        Validates: Requirements 4.3
        """
        operation_name = "convenience_test"
        fallback_func = Mock(return_value="convenience_result")
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'), \
             patch('torch.cuda.memory_allocated', return_value=1024*1024*1024), \
             patch('torch.cuda.memory_reserved', return_value=2*1024*1024*1024):
            
            # Test convenience function
            result = handle_gpu_memory_error(operation_name, fallback_func)
            
            # Should return fallback result
            assert result == "convenience_result", "Convenience function should return fallback result"
            
            # Should call fallback function
            fallback_func.assert_called_once()


if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v", "--tb=short"])