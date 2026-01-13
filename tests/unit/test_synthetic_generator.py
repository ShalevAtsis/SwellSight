"""
Unit tests for synthetic data generation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from swellsight.core.synthetic_generator import (
    FLUXControlNetGenerator,
    WeatherConditions,
    GenerationConfig,
    SyntheticImage,
    create_default_weather_conditions,
    validate_generation_config,
    estimate_generation_time,
    DIFFUSERS_AVAILABLE
)
from swellsight.core.depth_extractor import DepthMap


class TestWeatherConditions:
    """Test WeatherConditions dataclass."""
    
    def test_weather_conditions_creation(self):
        """Test creating weather conditions."""
        conditions = WeatherConditions(
            lighting="sunny",
            weather="clear", 
            wind_strength=0.5,
            wave_foam=0.3
        )
        
        assert conditions.lighting == "sunny"
        assert conditions.weather == "clear"
        assert conditions.wind_strength == 0.5
        assert conditions.wave_foam == 0.3


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        
        assert config.resolution == (1024, 1024)
        assert 0.3 <= config.controlnet_conditioning_scale <= 0.7
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 7.5
        assert config.seed is None
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = GenerationConfig(controlnet_conditioning_scale=0.5)
        # Should not raise
        
        # Invalid conditioning scale
        with pytest.raises(ValueError, match="controlnet_conditioning_scale must be between 0.3 and 0.7"):
            GenerationConfig(controlnet_conditioning_scale=0.2)
            
        with pytest.raises(ValueError, match="controlnet_conditioning_scale must be between 0.3 and 0.7"):
            GenerationConfig(controlnet_conditioning_scale=0.8)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_default_weather_conditions(self):
        """Test creating default weather conditions."""
        conditions = create_default_weather_conditions()
        
        assert len(conditions) > 0
        assert all(isinstance(c, WeatherConditions) for c in conditions)
        
        # Check we have diverse conditions
        lightings = {c.lighting for c in conditions}
        weathers = {c.weather for c in conditions}
        
        assert len(lightings) > 1
        assert len(weathers) > 1
        
    def test_validate_generation_config(self):
        """Test configuration validation function."""
        # Valid config
        config = GenerationConfig()
        validate_generation_config(config)  # Should not raise
        
        # Invalid resolution
        config.resolution = (0, 1024)
        with pytest.raises(ValueError, match="Invalid resolution"):
            validate_generation_config(config)
            
        # Invalid conditioning scale
        config = GenerationConfig()
        config.controlnet_conditioning_scale = 1.5
        with pytest.raises(ValueError, match="controlnet_conditioning_scale must be between"):
            validate_generation_config(config)
            
        # Invalid inference steps
        config = GenerationConfig()
        config.num_inference_steps = -1
        with pytest.raises(ValueError, match="num_inference_steps must be positive"):
            validate_generation_config(config)
            
    def test_estimate_generation_time(self):
        """Test generation time estimation."""
        config = GenerationConfig()
        
        # GPU should be faster than CPU
        gpu_time = estimate_generation_time(config, "cuda")
        cpu_time = estimate_generation_time(config, "cpu")
        
        assert gpu_time > 0
        assert cpu_time > 0
        assert cpu_time > gpu_time
        
        # More steps should take longer
        config_more_steps = GenerationConfig(num_inference_steps=100)
        longer_time = estimate_generation_time(config_more_steps, "cuda")
        
        assert longer_time > gpu_time


@pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="Diffusers not available")
class TestFLUXControlNetGenerator:
    """Test FLUX ControlNet generator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = FLUXControlNetGenerator()
        
        assert generator.model_path == "black-forest-labs/FLUX.1-dev"
        assert generator.controlnet_path == "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"
        assert generator._pipeline is None
        assert generator._controlnet is None
        
    def test_init_custom_paths(self):
        """Test generator initialization with custom paths."""
        generator = FLUXControlNetGenerator(
            model_path="custom/flux-model",
            controlnet_path="custom/controlnet-model"
        )
        
        assert generator.model_path == "custom/flux-model"
        assert generator.controlnet_path == "custom/controlnet-model"
        
    def test_prepare_depth_image(self):
        """Test depth image preparation."""
        generator = FLUXControlNetGenerator()
        
        # Create mock depth map
        depth_data = np.random.rand(512, 512).astype(np.float32)
        depth_map = DepthMap(
            data=depth_data,
            resolution=(512, 512),
            quality_score=0.8,
            edge_preservation=0.7
        )
        
        # Prepare depth image
        depth_image = generator._prepare_depth_image(depth_map)
        
        assert isinstance(depth_image, Image.Image)
        assert depth_image.size == (1024, 1024)  # Should be resized
        assert depth_image.mode == "RGB"
        
    def test_create_weather_prompt(self):
        """Test weather prompt creation."""
        generator = FLUXControlNetGenerator()
        
        conditions = WeatherConditions(
            lighting="sunny",
            weather="clear",
            wind_strength=0.8,
            wave_foam=0.6
        )
        
        base_prompt = "Ocean waves"
        enhanced_prompt = generator._create_weather_prompt(conditions, base_prompt)
        
        assert "Ocean waves" in enhanced_prompt
        assert "sunny" in enhanced_prompt.lower() or "bright" in enhanced_prompt.lower()
        assert "clear" in enhanced_prompt.lower()
        
    def test_extract_wave_metrics(self):
        """Test wave metrics extraction."""
        generator = FLUXControlNetGenerator()
        
        # Create mock depth map with some variation
        depth_data = np.random.rand(256, 256).astype(np.float32)
        depth_data[100:150, 100:150] += 0.2  # Add some depth variation
        
        depth_map = DepthMap(
            data=depth_data,
            resolution=(256, 256),
            quality_score=0.8,
            edge_preservation=0.7
        )
        
        conditions = WeatherConditions("sunny", "clear", 0.5, 0.4)
        config = GenerationConfig()
        
        metrics = generator._extract_wave_metrics(depth_map, conditions, config)
        
        assert 0.5 <= metrics.height_meters <= 8.0
        assert metrics.height_feet == metrics.height_meters * 3.28084
        assert metrics.direction in ["LEFT", "RIGHT", "STRAIGHT"]
        assert metrics.breaking_type in ["SPILLING", "PLUNGING", "SURGING"]
        assert 0.0 <= metrics.height_confidence <= 1.0
        assert 0.0 <= metrics.direction_confidence <= 1.0
        assert 0.0 <= metrics.breaking_confidence <= 1.0
        assert isinstance(metrics.extreme_conditions, bool)
        
    @patch('swellsight.core.synthetic_generator.FluxControlNetPipeline')
    @patch('swellsight.core.synthetic_generator.FluxControlNetModel')
    def test_generate_wave_scene_mock(self, mock_controlnet_model, mock_pipeline_class):
        """Test wave scene generation with mocked pipeline."""
        # Setup mocks
        mock_controlnet = Mock()
        mock_controlnet_model.from_pretrained.return_value = mock_controlnet
        
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        
        # Mock generation result
        mock_result_image = Image.new('RGB', (1024, 1024), color='blue')
        mock_result = Mock()
        mock_result.images = [mock_result_image]
        mock_pipeline.return_value = mock_result
        
        # Create generator and test
        generator = FLUXControlNetGenerator()
        
        depth_data = np.random.rand(256, 256).astype(np.float32)
        depth_map = DepthMap(
            data=depth_data,
            resolution=(256, 256),
            quality_score=0.8,
            edge_preservation=0.7
        )
        
        conditions = WeatherConditions("sunny", "clear", 0.5, 0.4)
        config = GenerationConfig(seed=42)
        
        result = generator.generate_wave_scene(depth_map, conditions, config)
        
        assert isinstance(result, SyntheticImage)
        assert result.rgb_data.shape == (1024, 1024, 3)
        assert result.depth_map == depth_map
        assert result.generation_params == config
        assert isinstance(result.ground_truth_labels.height_meters, float)
        
        # Verify pipeline was called correctly
        mock_pipeline_class.from_pretrained.assert_called_once()
        mock_pipeline.assert_called_once()
        
    def test_cleanup(self):
        """Test resource cleanup."""
        generator = FLUXControlNetGenerator()
        
        # Mock pipeline and controlnet
        generator._pipeline = Mock()
        generator._controlnet = Mock()
        
        generator.cleanup()
        
        assert generator._pipeline is None
        assert generator._controlnet is None


    @patch('swellsight.core.synthetic_generator.FluxControlNetPipeline')
    @patch('swellsight.core.synthetic_generator.FluxControlNetModel')
    def test_create_balanced_dataset_mock(self, mock_controlnet_model, mock_pipeline_class):
        """Test balanced dataset creation with mocked pipeline."""
        # Setup mocks
        mock_controlnet = Mock()
        mock_controlnet_model.from_pretrained.return_value = mock_controlnet
        
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        
        # Mock generation result
        mock_result_image = Image.new('RGB', (1024, 1024), color='blue')
        mock_result = Mock()
        mock_result.images = [mock_result_image]
        mock_pipeline.return_value = mock_result
        
        # Create generator and test
        generator = FLUXControlNetGenerator()
        
        # Test small dataset for speed
        target_size = 8
        dataset = generator.create_balanced_dataset(target_size)
        
        # Verify dataset structure
        assert hasattr(dataset, 'images')
        assert hasattr(dataset, 'balance_metrics')
        assert hasattr(dataset, 'statistics')
        
        # Check that we got some images (may be less than target due to validation)
        assert len(dataset.images) > 0
        assert len(dataset.images) <= target_size
        
        # Verify balance metrics structure
        assert 'height_distribution' in dataset.balance_metrics
        assert 'direction_distribution' in dataset.balance_metrics
        assert 'breaking_type_distribution' in dataset.balance_metrics
        assert 'weather_distribution' in dataset.balance_metrics
        
        # Verify statistics structure
        assert 'total_images' in dataset.statistics
        assert 'height_statistics' in dataset.statistics
        assert 'confidence_statistics' in dataset.statistics
        assert 'distribution_balance' in dataset.statistics
        
    def test_create_balanced_dataset_invalid_size(self):
        """Test balanced dataset creation with invalid target size."""
        generator = FLUXControlNetGenerator()
        
        with pytest.raises(ValueError, match="target_size must be positive"):
            generator.create_balanced_dataset(0)
            
        with pytest.raises(ValueError, match="target_size must be positive"):
            generator.create_balanced_dataset(-5)
    
    def test_generate_target_depth_map(self):
        """Test target depth map generation."""
        generator = FLUXControlNetGenerator()
        
        conditions = WeatherConditions("sunny", "clear", 0.5, 0.4)
        
        # Test different wave characteristics
        depth_map = generator._generate_target_depth_map(
            target_height_range=(1.0, 2.0),
            target_direction="RIGHT",
            target_breaking="PLUNGING",
            conditions=conditions,
            resolution=(128, 128)
        )
        
        assert depth_map.data.shape == (128, 128)
        assert depth_map.resolution == (128, 128)
        assert 0.0 <= np.min(depth_map.data) <= 1.0
        assert 0.0 <= np.max(depth_map.data) <= 1.0
        assert depth_map.quality_score > 0.0
        assert depth_map.edge_preservation > 0.0
        
    def test_calculate_balance_score(self):
        """Test balance score calculation."""
        generator = FLUXControlNetGenerator()
        
        # Perfect balance
        perfect_dist = {'A': 10, 'B': 10, 'C': 10}
        perfect_score = generator._calculate_balance_score(perfect_dist)
        assert perfect_score > 0.9  # Should be close to 1.0
        
        # Unbalanced distribution
        unbalanced_dist = {'A': 20, 'B': 1, 'C': 1}
        unbalanced_score = generator._calculate_balance_score(unbalanced_dist)
        assert unbalanced_score < 0.5  # Should be much lower
        
        # Empty distribution
        empty_dist = {}
        empty_score = generator._calculate_balance_score(empty_dist)
        assert empty_score == 0.0
        
    def test_validate_dataset_balance(self):
        """Test dataset balance validation."""
        generator = FLUXControlNetGenerator()
        
        # Create mock balance metrics
        balance_metrics = {
            'height_distribution': {'0.5-1.5m': 5, '1.5-3.0m': 5, '3.0-5.0m': 5, '5.0-8.0m': 5},
            'direction_distribution': {'LEFT': 7, 'RIGHT': 7, 'STRAIGHT': 6},
            'breaking_type_distribution': {'SPILLING': 7, 'PLUNGING': 7, 'SURGING': 6},
            'weather_distribution': {'sunny_clear': 10, 'overcast_fog': 10}
        }
        
        validation = generator._validate_dataset_balance(balance_metrics, 20)
        
        assert 'is_balanced' in validation
        assert 'balance_issues' in validation
        assert 'recommendations' in validation
        assert 'balance_scores' in validation
        
        # Check balance scores
        assert 'height' in validation['balance_scores']
        assert 'direction' in validation['balance_scores']
        assert 'breaking_type' in validation['balance_scores']
        assert 'weather' in validation['balance_scores']
        
    def test_validate_synthetic_vs_real_distribution(self):
        """Test synthetic vs real distribution validation."""
        generator = FLUXControlNetGenerator()
        
        # Create mock synthetic dataset
        mock_dataset = Mock()
        mock_dataset.statistics = {
            'height_statistics': {
                'mean': 2.3,
                'std': 1.1,
                'min': 0.6,
                'max': 6.8,
                'median': 2.1
            },
            'confidence_statistics': {
                'height_confidence': {'mean': 0.83, 'std': 0.13, 'min': 0.58},
                'direction_confidence': {'mean': 0.80, 'std': 0.16, 'min': 0.48},
                'breaking_confidence': {'mean': 0.78, 'std': 0.19, 'min': 0.38}
            },
            'total_images': 100,
            'extreme_conditions_count': 12
        }
        
        # Test with default expected statistics
        validation = generator.validate_synthetic_vs_real_distribution(mock_dataset)
        
        assert 'distribution_match' in validation
        assert 'comparison_metrics' in validation
        assert 'warnings' in validation
        assert 'recommendations' in validation
        assert 'overall_similarity_score' in validation
        
        # Should use expected statistics and warn about it
        assert any("Using expected statistics" in warning for warning in validation['warnings'])
        
    def test_compare_distributions(self):
        """Test distribution comparison."""
        generator = FLUXControlNetGenerator()
        
        # Similar distributions
        synthetic_stats = {'mean': 2.5, 'std': 1.2}
        real_stats = {'mean': 2.4, 'std': 1.1}
        
        comparison = generator._compare_distributions(synthetic_stats, real_stats, 'height')
        
        assert comparison['is_similar'] is True
        assert comparison['similarity_score'] > 0.8
        assert 'mean_relative_diff' in comparison['differences']
        assert 'std_relative_diff' in comparison['differences']
        
        # Very different distributions
        synthetic_stats = {'mean': 5.0, 'std': 2.0}
        real_stats = {'mean': 2.0, 'std': 1.0}
        
        comparison = generator._compare_distributions(synthetic_stats, real_stats, 'height')
        
        assert comparison['is_similar'] is False
        assert comparison['similarity_score'] < 0.5
        assert comparison['reason'] != ''
        
    def test_calculate_dataset_statistics(self):
        """Test dataset statistics calculation."""
        generator = FLUXControlNetGenerator()
        
        # Create mock synthetic images
        mock_images = []
        for i in range(5):
            mock_image = Mock()
            mock_image.ground_truth_labels = Mock()
            mock_image.ground_truth_labels.height_meters = 1.5 + i * 0.5
            mock_image.ground_truth_labels.direction = ["LEFT", "RIGHT", "STRAIGHT"][i % 3]
            mock_image.ground_truth_labels.breaking_type = ["SPILLING", "PLUNGING", "SURGING"][i % 3]
            mock_image.ground_truth_labels.height_confidence = 0.8 + i * 0.02
            mock_image.ground_truth_labels.direction_confidence = 0.75 + i * 0.03
            mock_image.ground_truth_labels.breaking_confidence = 0.7 + i * 0.04
            mock_image.ground_truth_labels.extreme_conditions = i > 3
            mock_image.generation_params = Mock()
            mock_image.generation_params.seed = 42 + i
            mock_image.generation_params.controlnet_conditioning_scale = 0.4 + i * 0.05
            mock_images.append(mock_image)
        
        # Create mock balance metrics
        balance_metrics = {
            'height_distribution': {'1.0-2.0m': 2, '2.0-3.0m': 3},
            'direction_distribution': {'LEFT': 2, 'RIGHT': 2, 'STRAIGHT': 1},
            'breaking_type_distribution': {'SPILLING': 2, 'PLUNGING': 2, 'SURGING': 1},
            'weather_distribution': {'sunny_clear': 5}
        }
        
        statistics = generator._calculate_dataset_statistics(mock_images, balance_metrics)
        
        assert statistics['total_images'] == 5
        assert 'height_statistics' in statistics
        assert 'confidence_statistics' in statistics
        assert 'distribution_balance' in statistics
        assert 'extreme_conditions_count' in statistics
        assert 'generation_parameters' in statistics
        
        # Check specific values
        assert statistics['height_statistics']['mean'] == 2.5  # (1.5+2.0+2.5+3.0+3.5)/5
        assert statistics['extreme_conditions_count'] == 1  # Only last image has extreme conditions


class TestSyntheticGeneratorWithoutDiffusers:
    """Test behavior when diffusers is not available."""
    
    @patch('swellsight.core.synthetic_generator.DIFFUSERS_AVAILABLE', False)
    def test_flux_generator_without_diffusers(self):
        """Test that FLUXControlNetGenerator raises error without diffusers."""
        with pytest.raises(ImportError, match="Diffusers library is required"):
            FLUXControlNetGenerator()