#!/usr/bin/env python3
"""
Property-Based Test for Dynamic Batch Sizing Adaptation
Feature: swellsight-pipeline-improvements, Property 6: Dynamic Batch Sizing Adaptation
Validates: Requirements 2.1
"""

import sys
import pytest
from hypothesis import given, strategies as st, settings, example, assume

# Add utils to path
sys.path.append('.')

from utils.memory_optimizer import MemoryOptimizer

class TestBatchSizeAdaptation:
    """Property-based tests for dynamic batch sizing adaptation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = MemoryOptimizer(safety_margin=0.1)
    
    @given(
        available_memory=st.integers(min_value=1024*1024*100, max_value=1024*1024*1024*16),  # 100MB to 16GB
        item_size=st.integers(min_value=1024, max_value=1024*1024*100),  # 1KB to 100MB
        max_batch_size=st.integers(min_value=1, max_value=128)
    )
    @settings(max_examples=100, deadline=3000)
    @example(available_memory=1024*1024*1024, item_size=1024*1024*10, max_batch_size=32)
    def test_batch_size_adaptation_property(self, available_memory: int, item_size: int, max_batch_size: int):
        """
        Property: For any available memory configuration, batch sizes should be 
        dynamically adjusted to optimize memory usage without exceeding limits
        
        Feature: swellsight-pipeline-improvements, Property 6: Dynamic Batch Sizing Adaptation
        Validates: Requirements 2.1
        """
        # Ensure reasonable constraints
        assume(available_memory >= item_size)  # Must be able to fit at least one item
        assume(item_size > 0)
        assume(max_batch_size > 0)
        
        # Calculate optimal batch size
        batch_size = self.optimizer.get_optimal_batch_size(
            available_memory=available_memory,
            item_size=item_size,
            max_batch_size=max_batch_size
        )
        
        # Property assertions
        assert isinstance(batch_size, int), "Batch size must be an integer"
        assert batch_size >= 1, f"Batch size must be at least 1, got {batch_size}"
        assert batch_size <= max_batch_size, f"Batch size {batch_size} must not exceed max {max_batch_size}"
        
        # Memory constraint: batch should not exceed available memory (with safety margin)
        usable_memory = int(available_memory * (1.0 - self.optimizer.safety_margin))  # Match implementation
        memory_used = batch_size * item_size
        
        # Special case: if item_size > usable_memory, the optimizer will still return 1
        # This is acceptable behavior for edge cases
        if item_size <= usable_memory:
            assert memory_used <= usable_memory, \
                f"Batch memory usage {memory_used} should not exceed usable memory {usable_memory}"
        else:
            # Edge case: item is larger than usable memory, but we still need batch_size >= 1
            assert batch_size == 1, f"When item is larger than usable memory, should return batch size 1"
        
        # Efficiency property: should use memory efficiently
        # If we can fit more items within memory limits, batch size should be larger
        theoretical_max = usable_memory // item_size  # Use integer division like implementation
        expected_batch = min(theoretical_max, max_batch_size)
        expected_batch = max(1, expected_batch)  # At least 1
        
        # Only check efficiency if item fits in memory
        if item_size <= usable_memory:
            assert batch_size == expected_batch, \
                f"Expected batch size {expected_batch}, got {batch_size} " \
                f"(available: {available_memory}, item_size: {item_size}, max: {max_batch_size})"
    
    @given(
        max_batch_size=st.integers(min_value=1, max_value=64)
    )
    @settings(max_examples=50, deadline=2000)
    def test_memory_constraint_property(self, max_batch_size: int):
        """
        Property: For any memory constraint, the calculated batch size should 
        respect both memory limits and maximum batch size
        
        Feature: swellsight-pipeline-improvements, Property 6: Dynamic Batch Sizing Adaptation
        Validates: Requirements 2.1
        """
        # Test with very limited memory
        small_memory = 1024 * 1024 * 10  # 10MB
        large_item_size = 1024 * 1024 * 5  # 5MB per item
        
        batch_size = self.optimizer.get_optimal_batch_size(
            available_memory=small_memory,
            item_size=large_item_size,
            max_batch_size=max_batch_size
        )
        
        # Should be constrained by memory, not max_batch_size
        usable_memory = int(small_memory * (1.0 - self.optimizer.safety_margin))  # Match implementation
        max_items_by_memory = usable_memory // large_item_size  # Use integer division
        expected_batch = min(max_items_by_memory, max_batch_size)
        expected_batch = max(1, expected_batch)
        
        assert batch_size == expected_batch, \
            f"Expected batch size {expected_batch}, got {batch_size}"
        assert batch_size >= 1, "Should always return at least batch size 1"
    
    @given(
        available_memory=st.integers(min_value=1024*1024*1024, max_value=1024*1024*1024*8),  # 1GB to 8GB
        item_size=st.integers(min_value=1024, max_value=1024*1024)  # 1KB to 1MB
    )
    @settings(max_examples=50, deadline=2000)
    def test_large_memory_property(self, available_memory: int, item_size: int):
        """
        Property: For any large memory configuration with small items,
        batch size should be constrained by max_batch_size, not memory
        
        Feature: swellsight-pipeline-improvements, Property 6: Dynamic Batch Sizing Adaptation
        Validates: Requirements 2.1
        """
        max_batch_size = 32  # Reasonable limit
        
        batch_size = self.optimizer.get_optimal_batch_size(
            available_memory=available_memory,
            item_size=item_size,
            max_batch_size=max_batch_size
        )
        
        # With large memory and small items, should hit max_batch_size limit
        usable_memory = int(available_memory * (1.0 - self.optimizer.safety_margin))  # Match implementation
        theoretical_max = usable_memory // item_size  # Use integer division
        
        if theoretical_max >= max_batch_size:
            assert batch_size == max_batch_size, \
                f"With large memory, should hit max batch size limit: expected {max_batch_size}, got {batch_size}"
        else:
            assert batch_size == theoretical_max, \
                f"Should use all available memory: expected {theoretical_max}, got {batch_size}"
    
    def test_zero_memory_edge_case(self):
        """
        Property: For edge cases with very low memory or large items,
        should return minimum viable batch size
        
        Feature: swellsight-pipeline-improvements, Property 6: Dynamic Batch Sizing Adaptation
        Validates: Requirements 2.1
        """
        # Test with insufficient memory
        tiny_memory = 1024  # 1KB
        large_item = 1024 * 1024  # 1MB
        
        batch_size = self.optimizer.get_optimal_batch_size(
            available_memory=tiny_memory,
            item_size=large_item,
            max_batch_size=10
        )
        
        # Should return at least 1, even if memory is insufficient
        assert batch_size >= 1, "Should always return at least batch size 1"
        assert isinstance(batch_size, int), "Batch size must be an integer"
    
    def test_none_parameters_property(self):
        """
        Property: For any None parameters, should use reasonable defaults
        and return valid batch size
        
        Feature: swellsight-pipeline-improvements, Property 6: Dynamic Batch Sizing Adaptation
        Validates: Requirements 2.1
        """
        # Test with None parameters (should use auto-detection)
        batch_size = self.optimizer.get_optimal_batch_size(
            available_memory=None,
            item_size=None,
            max_batch_size=16
        )
        
        assert isinstance(batch_size, int), "Batch size must be an integer"
        assert batch_size >= 1, "Batch size must be at least 1"
        assert batch_size <= 16, "Batch size must respect max_batch_size"
    
    @given(
        safety_margin=st.floats(min_value=0.0, max_value=0.5)
    )
    @settings(max_examples=30, deadline=2000)
    def test_safety_margin_property(self, safety_margin: float):
        """
        Property: For any safety margin, the optimizer should respect it
        when calculating batch sizes
        
        Feature: swellsight-pipeline-improvements, Property 6: Dynamic Batch Sizing Adaptation
        Validates: Requirements 2.1
        """
        # Create optimizer with specific safety margin
        optimizer = MemoryOptimizer(safety_margin=safety_margin)
        
        available_memory = 1024 * 1024 * 100  # 100MB
        item_size = 1024 * 1024 * 10  # 10MB per item
        max_batch_size = 20
        
        batch_size = optimizer.get_optimal_batch_size(
            available_memory=available_memory,
            item_size=item_size,
            max_batch_size=max_batch_size
        )
        
        # Calculate expected batch size with safety margin
        usable_memory = int(available_memory * (1.0 - safety_margin))  # Match implementation
        expected_max = usable_memory // item_size  # Use integer division
        expected_batch = min(expected_max, max_batch_size)
        expected_batch = max(1, expected_batch)
        
        assert batch_size == expected_batch, \
            f"Safety margin {safety_margin} not respected: expected {expected_batch}, got {batch_size}"
        
        # Verify memory usage doesn't exceed limit
        memory_used = batch_size * item_size
        assert memory_used <= usable_memory, \
            f"Memory usage {memory_used} exceeds limit {usable_memory} with safety margin {safety_margin}"


if __name__ == "__main__":
    # Run the property tests
    pytest.main([__file__, "-v", "--tb=short"])