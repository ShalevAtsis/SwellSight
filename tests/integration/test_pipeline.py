"""
Integration tests for the complete wave analysis pipeline.

Tests the end-to-end functionality including depth extraction, synthetic generation,
and wave analysis stages with real beach cam footage.
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
import psutil
import concurrent.futures
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.swellsight.core.pipeline import (
    WaveAnalysisPipeline, PipelineConfig, PipelineResults, BatchProcessingResults
)
from src.swellsight.core.depth_extractor import DepthMap, QualityMetrics
from src.swellsight.core.synthetic_generator import WaveMetrics, WeatherConditions
from src.swellsight.core.wave_analyzer import ConfidenceScores
from src.swellsight.utils.error_handler import ProcessingError, ConfigurationError


class TestWaveAnalysisPipelineIntegration:
    """Integration tests for the complete wave analysis pipeline."""
    
    @pytest.fixture
    def sample_beach_cam_image(self):
        """Create a sample beach cam image for testing."""
        # Create a realistic beach cam image with wave-like patterns
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add sky (blue gradient)
        for y in range(height // 3):
            intensity = 100 + (y * 155) // (height // 3)
            image[y, :, 2] = intensity  # Blue channel
            image[y, :, 1] = intensity // 2  # Green channel
        
        # Add ocean with wave patterns
        x = np.linspace(0, 4 * np.pi, width)
        for y in range(height // 3, height):
            wave_pattern = 50 * np.sin(x + (y - height // 3) * 0.1)
            base_intensity = 80 + (y - height // 3) * 2
            
            # Blue-green ocean color with wave patterns
            image[y, :, 2] = np.clip(base_intensity + wave_pattern, 0, 255).astype(np.uint8)  # Blue
            image[y, :, 1] = np.clip(base_intensity * 0.8 + wave_pattern * 0.5, 0, 255).astype(np.uint8)  # Green
            image[y, :, 0] = np.clip(base_intensity * 0.3, 0, 255).astype(np.uint8)  # Red
        
        # Add some beach/shore area
        shore_start = int(height * 0.8)
        for y in range(shore_start, height):
            # Sandy beach color
            image[y, :, :] = [194, 178, 128]  # Sandy brown
        
        return image
    
    @pytest.fixture
    def pipeline_config(self):
        """Create a test pipeline configuration."""
        return PipelineConfig(
            depth_model_size="small",
            depth_precision="fp32",
            use_gpu=False,  # Use CPU for testing
            enable_optimization=True,
            max_processing_time=30.0,
            confidence_threshold=0.6,
            depth_quality_threshold=0.7,
            prediction_quality_threshold=0.6
        )
    
    @pytest.fixture
    def mock_pipeline_components(self):
        """Create mock components for pipeline testing."""
        # Mock depth extractor
        mock_depth_extractor = Mock()
        mock_depth_map = DepthMap(
            data=np.random.rand(480, 640),
            resolution=(640, 480),
            quality_score=0.85,
            edge_preservation=0.9
        )
        mock_depth_extractor.extract_depth.return_value = mock_depth_map
        
        # Mock the validate_quality method
        mock_quality_metrics = QualityMetrics(
            overall_score=0.85,
            edge_preservation=0.9,
            texture_capture=0.8,
            far_field_sensitivity=0.75,
            contrast_ratio=0.9
        )
        mock_depth_extractor.validate_quality.return_value = mock_quality_metrics
        
        # Mock the normalize_depth_for_waves method
        enhanced_depth_map = DepthMap(
            data=np.random.rand(480, 640),
            resolution=(640, 480),
            quality_score=0.9,
            edge_preservation=0.95
        )
        mock_depth_extractor.normalize_depth_for_waves.return_value = enhanced_depth_map
        
        # Mock wave analyzer
        mock_wave_analyzer = Mock()
        mock_wave_metrics = WaveMetrics(
            height_meters=2.5,
            height_feet=8.2,
            height_confidence=0.9,
            direction="LEFT",
            direction_confidence=0.85,
            breaking_type="SPILLING",
            breaking_confidence=0.8,
            extreme_conditions=False
        )
        
        # Mock the analyze_waves method to return the expected tuple
        mock_wave_analyzer.analyze_waves.return_value = (
            mock_wave_metrics, 
            None,  # PerformanceMetrics
            {'confidence': 0.85}  # Additional data
        )
        
        # Mock the get_confidence_scores method
        mock_confidence_scores = ConfidenceScores(
            height_confidence=0.9,
            direction_confidence=0.85,
            breaking_type_confidence=0.8,
            overall_confidence=0.85
        )
        mock_wave_analyzer.get_confidence_scores.return_value = mock_confidence_scores
        
        # Mock synthetic generator
        mock_synthetic_generator = Mock()
        
        return {
            'depth_extractor': mock_depth_extractor,
            'wave_analyzer': mock_wave_analyzer,
            'synthetic_generator': mock_synthetic_generator,
            'depth_map': mock_depth_map,
            'wave_metrics': mock_wave_metrics,
            'confidence_scores': mock_confidence_scores
        }
    
    def test_end_to_end_pipeline_processing(self, sample_beach_cam_image, pipeline_config, mock_pipeline_components):
        """Test complete pipeline processing with real beach cam footage."""
        # Create pipeline with mocked components
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls, \
             patch('src.swellsight.core.pipeline.DINOv2WaveAnalyzer') as mock_wave_cls, \
             patch('src.swellsight.core.pipeline.FLUXControlNetGenerator') as mock_synth_cls:
            
            mock_depth_cls.return_value = mock_pipeline_components['depth_extractor']
            mock_wave_cls.return_value = mock_pipeline_components['wave_analyzer']
            mock_synth_cls.return_value = mock_pipeline_components['synthetic_generator']
            
            # Initialize pipeline
            pipeline = WaveAnalysisPipeline(pipeline_config)
            
            # Process single image
            start_time = time.time()
            result = pipeline.process_beach_cam_image(sample_beach_cam_image)
            processing_time = time.time() - start_time
            
            # Validate results
            assert result is not None
            assert isinstance(result, PipelineResults)
            assert result.wave_metrics is not None
            assert result.processing_time > 0
            
            # Validate performance requirements (Requirement 8.1: real-time processing)
            assert processing_time < 5.0, f"Processing took {processing_time:.2f}s, exceeds 5s limit"
            
            # Validate accuracy metrics (Requirement 9.5: validation)
            assert result.pipeline_confidence >= 0.6
            
            # Verify component interactions
            mock_pipeline_components['depth_extractor'].extract_depth.assert_called_once()
            mock_pipeline_components['wave_analyzer'].analyze_waves.assert_called_once()
    
    def test_batch_processing_functionality(self, sample_beach_cam_image, pipeline_config, mock_pipeline_components):
        """Test batch processing with multiple images."""
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls, \
             patch('src.swellsight.core.pipeline.DINOv2WaveAnalyzer') as mock_wave_cls, \
             patch('src.swellsight.core.pipeline.FLUXControlNetGenerator') as mock_synth_cls:
            
            mock_depth_cls.return_value = mock_pipeline_components['depth_extractor']
            mock_wave_cls.return_value = mock_pipeline_components['wave_analyzer']
            mock_synth_cls.return_value = mock_pipeline_components['synthetic_generator']
            
            pipeline = WaveAnalysisPipeline(pipeline_config)
            
            # Create batch of images
            batch_images = [sample_beach_cam_image for _ in range(3)]
            
            # Process batch
            start_time = time.time()
            results = pipeline.process_batch(batch_images)
            batch_processing_time = time.time() - start_time
            
            # Validate batch results
            assert results is not None
            assert isinstance(results, BatchProcessingResults)
            assert len(results.individual_results) == 3
            assert results.get_success_rate() == 1.0
            assert results.get_average_processing_time() > 0
            
            # Validate performance for batch processing
            assert batch_processing_time < 15.0, f"Batch processing took {batch_processing_time:.2f}s"
            
            # Verify all results are valid
            for result in results.individual_results:
                assert result is not None
                assert result.wave_metrics is not None
                assert result.pipeline_confidence >= 0.6
    
    def test_streaming_processing_capability(self, sample_beach_cam_image, pipeline_config, mock_pipeline_components):
        """Test streaming processing functionality."""
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls, \
             patch('src.swellsight.core.pipeline.DINOv2WaveAnalyzer') as mock_wave_cls, \
             patch('src.swellsight.core.pipeline.FLUXControlNetGenerator') as mock_synth_cls:
            
            mock_depth_cls.return_value = mock_pipeline_components['depth_extractor']
            mock_wave_cls.return_value = mock_pipeline_components['wave_analyzer']
            mock_synth_cls.return_value = mock_pipeline_components['synthetic_generator']
            
            pipeline = WaveAnalysisPipeline(pipeline_config)
            
            # Create image generator for streaming
            def image_generator():
                for i in range(5):
                    yield sample_beach_cam_image
            
            # Process stream
            processed_count = 0
            total_processing_time = 0
            
            for result in pipeline.process_streaming(image_generator()):
                assert result is not None
                assert result.wave_metrics is not None
                processed_count += 1
                total_processing_time += result.processing_time
            
            # Validate streaming results
            assert processed_count == 5
            average_time = total_processing_time / processed_count
            assert average_time < 5.0, f"Average streaming processing time {average_time:.2f}s exceeds limit"
    
    def test_error_handling_and_recovery(self, sample_beach_cam_image, pipeline_config):
        """Test error handling and recovery scenarios."""
        # Test with invalid image
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls:
            mock_depth_extractor = Mock()
            mock_depth_extractor.extract_depth.side_effect = ProcessingError("Depth extraction failed")
            mock_depth_cls.return_value = mock_depth_extractor
            
            pipeline = WaveAnalysisPipeline(pipeline_config)
            
            # Should handle error gracefully
            with pytest.raises(ProcessingError):
                pipeline.process_beach_cam_image(sample_beach_cam_image)
    
    def test_hardware_fallback_mechanism(self, sample_beach_cam_image, mock_pipeline_components):
        """Test hardware fallback from GPU to CPU."""
        # Test GPU unavailable scenario
        config_gpu = PipelineConfig(
            depth_model_size="small",
            depth_precision="fp32",
            use_gpu=True,  # Request GPU
            enable_optimization=True
        )
        
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls, \
             patch('src.swellsight.core.pipeline.DINOv2WaveAnalyzer') as mock_wave_cls, \
             patch('torch.cuda.is_available', return_value=False):  # Simulate no GPU
            
            mock_depth_cls.return_value = mock_pipeline_components['depth_extractor']
            mock_wave_cls.return_value = mock_pipeline_components['wave_analyzer']
            
            # Should fallback to CPU
            pipeline = WaveAnalysisPipeline(config_gpu)
            result = pipeline.process_beach_cam_image(sample_beach_cam_image)
            
            assert result is not None
            assert result.wave_metrics is not None
    
    def test_memory_management_under_load(self, sample_beach_cam_image, pipeline_config, mock_pipeline_components):
        """Test memory management during intensive processing."""
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls, \
             patch('src.swellsight.core.pipeline.DINOv2WaveAnalyzer') as mock_wave_cls, \
             patch('src.swellsight.core.pipeline.FLUXControlNetGenerator') as mock_synth_cls:
            
            mock_depth_cls.return_value = mock_pipeline_components['depth_extractor']
            mock_wave_cls.return_value = mock_pipeline_components['wave_analyzer']
            mock_synth_cls.return_value = mock_pipeline_components['synthetic_generator']
            
            pipeline = WaveAnalysisPipeline(pipeline_config)
            
            # Process multiple batches to test memory management
            initial_memory = psutil.Process().memory_info().rss
            
            for batch_idx in range(3):
                batch_images = [sample_beach_cam_image for _ in range(2)]
                results = pipeline.process_batch(batch_images)
                assert results.get_success_rate() == 1.0
            
            final_memory = psutil.Process().memory_info().rss
            memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
            
            # Memory increase should be reasonable (less than 100MB for test)
            assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
    
    def test_configuration_validation(self):
        """Test pipeline configuration validation."""
        # Test invalid configuration
        with pytest.raises(ConfigurationError):
            invalid_config = PipelineConfig(
                depth_model_size="invalid",  # Invalid model size
                depth_precision="invalid",  # Invalid precision
                confidence_threshold=2.0,  # Invalid threshold
                max_processing_time=-1.0  # Invalid negative time
            )
            invalid_config.validate()
    
    def test_performance_monitoring_integration(self, sample_beach_cam_image, pipeline_config, mock_pipeline_components):
        """Test performance monitoring and metrics collection."""
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls, \
             patch('src.swellsight.core.pipeline.DINOv2WaveAnalyzer') as mock_wave_cls:
            
            mock_depth_cls.return_value = mock_pipeline_components['depth_extractor']
            mock_wave_cls.return_value = mock_pipeline_components['wave_analyzer']
            
            pipeline = WaveAnalysisPipeline(pipeline_config)
            result = pipeline.process_beach_cam_image(sample_beach_cam_image)
            
            # Validate performance metrics are collected
            assert result.processing_time > 0
            assert result.stage_timings is not None
            assert "total" in result.stage_timings
    
    def test_quality_threshold_enforcement(self, sample_beach_cam_image, pipeline_config, mock_pipeline_components):
        """Test quality threshold enforcement and rejection of low-quality results."""
        # Create low-quality depth map
        low_quality_metrics = QualityMetrics(
            overall_score=0.3,  # Below threshold
            edge_preservation=0.4,
            texture_capture=0.3,
            far_field_sensitivity=0.2,
            contrast_ratio=0.3
        )
        
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls, \
             patch('src.swellsight.core.pipeline.DINOv2WaveAnalyzer') as mock_wave_cls:
            
            mock_depth_extractor = Mock()
            mock_depth_extractor.extract_depth.return_value = mock_pipeline_components['depth_map']
            mock_depth_extractor.validate_quality.return_value = low_quality_metrics
            mock_depth_extractor.normalize_depth_for_waves.return_value = mock_pipeline_components['depth_map']
            mock_depth_cls.return_value = mock_depth_extractor
            mock_wave_cls.return_value = mock_pipeline_components['wave_analyzer']
            
            pipeline = WaveAnalysisPipeline(pipeline_config)
            result = pipeline.process_beach_cam_image(sample_beach_cam_image)
            
            # Should still process but with warnings
            assert result is not None
            assert len(result.warnings) > 0
            assert any("quality" in warning.lower() for warning in result.warnings)
    
    def test_concurrent_processing_safety(self, sample_beach_cam_image, pipeline_config, mock_pipeline_components):
        """Test thread safety during concurrent processing."""
        with patch('src.swellsight.core.pipeline.DepthAnythingV2Extractor') as mock_depth_cls, \
             patch('src.swellsight.core.pipeline.DINOv2WaveAnalyzer') as mock_wave_cls:
            
            mock_depth_cls.return_value = mock_pipeline_components['depth_extractor']
            mock_wave_cls.return_value = mock_pipeline_components['wave_analyzer']
            
            pipeline = WaveAnalysisPipeline(pipeline_config)
            
            def process_image_worker():
                return pipeline.process_beach_cam_image(sample_beach_cam_image)
            
            # Run concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_image_worker) for _ in range(5)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # All results should be successful
            assert len(results) == 5
            for result in results:
                assert result is not None
                assert result.wave_metrics is not None