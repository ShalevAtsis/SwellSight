"""
End-to-end Wave Analysis Pipeline

Orchestrates the three-stage pipeline: depth extraction, synthetic generation,
and wave analysis for complete beach cam footage processing.
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, asdict
import numpy as np
import logging
import time
import json
from pathlib import Path

from .depth_extractor import DepthExtractor, DepthAnythingV2Extractor, DepthMap, QualityMetrics
from .synthetic_generator import SyntheticDataGenerator, FLUXControlNetGenerator, WeatherConditions, GenerationConfig
from .wave_analyzer import WaveAnalyzer, DINOv2WaveAnalyzer, ConfidenceScores
from .synthetic_generator import WaveMetrics
from ..utils.performance import PerformanceMetrics
from ..utils.error_handler import (
    error_handler, retry_with_backoff, RetryConfig, safe_execute,
    ProcessingError, ConfigurationError, ErrorSeverity
)

@dataclass
class PipelineConfig:
    """Configuration for the complete wave analysis pipeline."""
    # Depth extraction configuration
    depth_model_size: str = "large"
    depth_precision: str = "fp16"
    depth_enhancement_factor: float = 2.0
    
    # Wave analysis configuration
    wave_backbone_model: str = "dinov2_vitb14"
    freeze_backbone: bool = True
    confidence_calibration_method: str = "isotonic"
    
    # Performance configuration
    use_gpu: bool = True
    enable_optimization: bool = True
    max_processing_time: float = 30.0  # seconds
    target_latency_ms: float = 200.0
    
    # Quality thresholds
    confidence_threshold: float = 0.7
    depth_quality_threshold: float = 0.5
    prediction_quality_threshold: float = 0.6
    
    # Error handling configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_fallback: bool = True
    
    # Output configuration
    save_intermediate_results: bool = False
    output_directory: Optional[str] = None
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.depth_model_size not in ["small", "base", "large"]:
            raise ConfigurationError(f"Invalid depth_model_size: {self.depth_model_size}")
        
        if self.depth_precision not in ["fp16", "fp32"]:
            raise ConfigurationError(f"Invalid depth_precision: {self.depth_precision}")
        
        if not (0.1 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError(f"confidence_threshold must be between 0.1 and 1.0")
        
        if self.max_processing_time <= 0:
            raise ConfigurationError("max_processing_time must be positive")
        
        if self.target_latency_ms <= 0:
            raise ConfigurationError("target_latency_ms must be positive")

@dataclass
class PipelineResults:
    """Results from complete pipeline execution."""
    wave_metrics: WaveMetrics
    processing_time: float
    pipeline_confidence: float
    stage_timings: Dict[str, float]
    warnings: List[str]
    
    # Enhanced results
    depth_quality: Optional[QualityMetrics] = None
    confidence_scores: Optional[ConfidenceScores] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    quality_validation: Optional[Dict[str, Any]] = None
    
    # Intermediate results (if enabled)
    depth_map: Optional[DepthMap] = None
    enhanced_depth_map: Optional[DepthMap] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        result_dict = asdict(self)
        
        # Handle numpy arrays and complex objects
        if self.depth_map is not None:
            result_dict['depth_map'] = {
                'resolution': self.depth_map.resolution,
                'quality_score': self.depth_map.quality_score,
                'edge_preservation': self.depth_map.edge_preservation
            }
        
        if self.enhanced_depth_map is not None:
            result_dict['enhanced_depth_map'] = {
                'resolution': self.enhanced_depth_map.resolution,
                'quality_score': self.enhanced_depth_map.quality_score,
                'edge_preservation': self.enhanced_depth_map.edge_preservation
            }
        
        return result_dict
    
    def save_to_file(self, filepath: str) -> None:
        """Save results to JSON file."""
        result_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

@dataclass
class BatchProcessingResults:
    """Results from batch processing multiple images."""
    individual_results: List[Optional[PipelineResults]]
    batch_statistics: Dict[str, Any]
    processing_summary: Dict[str, Any]
    failed_indices: List[int]
    
    def get_success_rate(self) -> float:
        """Get success rate for batch processing."""
        successful = sum(1 for result in self.individual_results if result is not None)
        return successful / len(self.individual_results) if self.individual_results else 0.0
    
    def get_average_processing_time(self) -> float:
        """Get average processing time for successful results."""
        successful_times = [
            result.processing_time for result in self.individual_results 
            if result is not None
        ]
        return np.mean(successful_times) if successful_times else 0.0

class WaveAnalysisPipeline:
    """Complete end-to-end wave analysis pipeline with comprehensive orchestration."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the wave analysis pipeline.
        
        Args:
            config: Pipeline configuration parameters
        """
        self.config = config or PipelineConfig()
        self.config.validate()  # Validate configuration
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline components
        self.depth_extractor: Optional[DepthExtractor] = None
        self.synthetic_generator: Optional[SyntheticDataGenerator] = None
        self.wave_analyzer: Optional[WaveAnalyzer] = None
        
        # Performance tracking
        self._processing_history: List[Dict[str, Any]] = []
        self._error_count = 0
        self._last_error_time = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components with comprehensive error handling."""
        try:
            self.logger.info("Initializing pipeline components...")
            
            # Stage A: Depth Extraction
            self.depth_extractor = DepthAnythingV2Extractor(
                model_size=self.config.depth_model_size,
                precision=self.config.depth_precision,
                enable_optimization=self.config.enable_optimization
            )
            
            # Stage B: Synthetic Data Generation (for training)
            if self.config.enable_optimization:
                try:
                    self.synthetic_generator = FLUXControlNetGenerator()
                    self.logger.info("Synthetic generator initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize synthetic generator: {e}")
                    self.synthetic_generator = None
            
            # Stage C: Wave Analysis
            self.wave_analyzer = DINOv2WaveAnalyzer(
                backbone_model=self.config.wave_backbone_model,
                freeze_backbone=self.config.freeze_backbone,
                enable_optimization=self.config.enable_optimization,
                confidence_calibration_method=self.config.confidence_calibration_method
            )
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise ConfigurationError(
                f"Pipeline initialization failed: {str(e)}",
                component="WaveAnalysisPipeline",
                operation="initialization",
                recovery_suggestions=[
                    "Check GPU availability and memory",
                    "Verify model dependencies are installed",
                    "Try reducing model sizes in configuration",
                    "Enable fallback mode in configuration"
                ]
            ) from e
    
    @retry_with_backoff(
        retry_config=RetryConfig(max_attempts=3, base_delay=1.0),
        exceptions=(ProcessingError, RuntimeError),
        component="WaveAnalysisPipeline",
        operation="image_processing"
    )
    def process_beach_cam_image(self, image: np.ndarray, save_intermediates: bool = None) -> PipelineResults:
        """Process a single beach cam image through the complete pipeline.
        
        Args:
            image: RGB beach cam image as numpy array
            save_intermediates: Override config setting for saving intermediate results
            
        Returns:
            PipelineResults with wave metrics and comprehensive performance data
        """
        start_time = time.time()
        stage_timings = {}
        warnings = []
        
        # Use config setting if not overridden
        if save_intermediates is None:
            save_intermediates = self.config.save_intermediate_results
        
        try:
            self.logger.info("Starting end-to-end wave analysis pipeline")
            
            # Stage A: Extract depth map
            stage_start = time.time()
            depth_result = self.depth_extractor.extract_depth(image)
            
            # Handle different return formats (with or without performance metrics)
            if isinstance(depth_result, tuple):
                depth_map, depth_performance = depth_result
            else:
                depth_map, depth_performance = depth_result, None
            
            stage_timings["depth_extraction"] = time.time() - stage_start
            
            # Validate depth quality
            depth_quality = self.depth_extractor.validate_quality(depth_map)
            if depth_quality.overall_score < self.config.depth_quality_threshold:
                warnings.append(f"Low depth quality: {depth_quality.overall_score:.2f}")
                self.logger.warning(f"Depth quality below threshold: {depth_quality.overall_score:.2f}")
            
            # Enhance depth map for better wave-ocean contrast
            stage_start = time.time()
            enhanced_depth_map = self.depth_extractor.normalize_depth_for_waves(
                depth_map, self.config.depth_enhancement_factor
            )
            stage_timings["depth_enhancement"] = time.time() - stage_start
            
            # Stage C: Analyze waves (Stage B is for training data generation)
            stage_start = time.time()
            wave_result = self.wave_analyzer.analyze_waves(image, enhanced_depth_map)
            
            # Handle different return formats
            if isinstance(wave_result, tuple) and len(wave_result) == 3:
                wave_metrics, wave_performance, quality_validation = wave_result
            elif isinstance(wave_result, tuple) and len(wave_result) == 2:
                wave_metrics, wave_performance = wave_result
                quality_validation = None
            else:
                wave_metrics, wave_performance, quality_validation = wave_result, None, None
            
            stage_timings["wave_analysis"] = time.time() - stage_start
            
            # Get confidence scores
            confidence_scores = self.wave_analyzer.get_confidence_scores()
            
            # Validate prediction quality
            if quality_validation and not quality_validation.get("prediction_validation", {}).get("is_valid", True):
                warnings.append("Anomalous prediction detected")
                self.logger.warning("Quality validation flagged anomalous prediction")
            
            # Calculate overall processing time
            total_time = time.time() - start_time
            stage_timings["total"] = total_time
            
            # Check processing time constraint
            if total_time > self.config.max_processing_time:
                warnings.append(f"Processing exceeded time limit: {total_time:.1f}s")
                self.logger.warning(f"Processing time exceeded limit: {total_time:.1f}s > {self.config.max_processing_time}s")
            
            # Check real-time performance requirement
            if total_time * 1000 > self.config.target_latency_ms:
                warnings.append(f"Processing exceeded real-time target: {total_time*1000:.1f}ms")
            
            # Create comprehensive results
            results = PipelineResults(
                wave_metrics=wave_metrics,
                processing_time=total_time,
                pipeline_confidence=confidence_scores.overall_confidence,
                stage_timings=stage_timings,
                warnings=warnings,
                depth_quality=depth_quality,
                confidence_scores=confidence_scores,
                performance_metrics=wave_performance or depth_performance,
                quality_validation=quality_validation,
                depth_map=depth_map if save_intermediates else None,
                enhanced_depth_map=enhanced_depth_map if save_intermediates else None
            )
            
            # Save results if configured
            if save_intermediates and self.config.output_directory:
                self._save_intermediate_results(results, image)
            
            # Update processing history
            self._update_processing_history(results)
            
            self.logger.info(f"Pipeline processing completed successfully in {total_time:.2f}s")
            self.logger.info(f"Wave metrics: {wave_metrics.height_meters:.1f}m, {wave_metrics.direction}, {wave_metrics.breaking_type}")
            
            return results
            
        except Exception as e:
            self._error_count += 1
            self._last_error_time = time.time()
            
            self.logger.error(f"Pipeline processing failed: {e}")
            
            # Try to provide partial results if possible
            if 'depth_map' in locals():
                warnings.append(f"Processing failed at wave analysis stage: {str(e)}")
                # Return partial results with default wave metrics
                default_wave_metrics = WaveMetrics(
                    height_meters=0.0,
                    height_feet=0.0,
                    height_confidence=0.0,
                    direction="STRAIGHT",
                    direction_confidence=0.0,
                    breaking_type="NO_BREAKING",
                    breaking_confidence=0.0,
                    extreme_conditions=False
                )
                
                return PipelineResults(
                    wave_metrics=default_wave_metrics,
                    processing_time=time.time() - start_time,
                    pipeline_confidence=0.0,
                    stage_timings=stage_timings,
                    warnings=warnings + [f"Pipeline failed: {str(e)}"],
                    depth_quality=locals().get('depth_quality'),
                    confidence_scores=None,
                    performance_metrics=None,
                    quality_validation=None
                )
            
            raise ProcessingError(
                f"Pipeline processing failed: {str(e)}",
                component="WaveAnalysisPipeline",
                operation="image_processing",
                recovery_suggestions=[
                    "Check input image format and quality",
                    "Verify GPU memory availability",
                    "Try reducing image resolution",
                    "Enable fallback processing mode"
                ]
            ) from e
    
    def process_batch(self, images: List[np.ndarray], progress_callback=None) -> BatchProcessingResults:
        """Process multiple beach cam images in batch with comprehensive tracking.
        
        Args:
            images: List of RGB beach cam images
            progress_callback: Optional callback function for progress updates
            
        Returns:
            BatchProcessingResults with detailed batch processing information
        """
        if not images:
            raise ValueError("Empty image list provided")
        
        self.logger.info(f"Starting batch processing for {len(images)} images")
        
        batch_start_time = time.time()
        results = []
        failed_indices = []
        processing_times = []
        confidence_scores = []
        
        for i, image in enumerate(images):
            try:
                self.logger.debug(f"Processing image {i+1}/{len(images)}")
                
                result = self.process_beach_cam_image(image)
                results.append(result)
                processing_times.append(result.processing_time)
                confidence_scores.append(result.pipeline_confidence)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(images), result)
                
                self.logger.debug(f"Successfully processed image {i+1}/{len(images)}")
                
            except Exception as e:
                self.logger.error(f"Failed to process image {i+1}: {e}")
                results.append(None)
                failed_indices.append(i)
                
                # Progress callback for failures
                if progress_callback:
                    progress_callback(i + 1, len(images), None)
        
        # Calculate batch statistics
        successful_results = [r for r in results if r is not None]
        batch_processing_time = time.time() - batch_start_time
        
        batch_statistics = {
            'total_images': len(images),
            'successful_images': len(successful_results),
            'failed_images': len(failed_indices),
            'success_rate': len(successful_results) / len(images),
            'average_processing_time': np.mean(processing_times) if processing_times else 0.0,
            'total_batch_time': batch_processing_time,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'throughput_images_per_second': len(images) / batch_processing_time
        }
        
        # Processing summary
        processing_summary = {
            'batch_start_time': batch_start_time,
            'batch_end_time': time.time(),
            'failed_indices': failed_indices,
            'performance_metrics': {
                'min_processing_time': min(processing_times) if processing_times else 0.0,
                'max_processing_time': max(processing_times) if processing_times else 0.0,
                'std_processing_time': np.std(processing_times) if processing_times else 0.0
            }
        }
        
        self.logger.info(f"Batch processing completed: {len(successful_results)}/{len(images)} successful")
        self.logger.info(f"Success rate: {batch_statistics['success_rate']:.1%}")
        self.logger.info(f"Average processing time: {batch_statistics['average_processing_time']:.2f}s")
        
        return BatchProcessingResults(
            individual_results=results,
            batch_statistics=batch_statistics,
            processing_summary=processing_summary,
            failed_indices=failed_indices
        )
    
    def process_streaming(self, image_generator, max_images: Optional[int] = None):
        """Process streaming beach cam images with real-time analysis.
        
        Args:
            image_generator: Generator yielding RGB beach cam images
            max_images: Maximum number of images to process (None for unlimited)
            
        Yields:
            PipelineResults for each processed image
        """
        self.logger.info("Starting streaming wave analysis")
        
        processed_count = 0
        
        for image in image_generator:
            if max_images and processed_count >= max_images:
                break
            
            try:
                result = self.process_beach_cam_image(image)
                yield result
                processed_count += 1
                
                # Check real-time performance
                if result.processing_time * 1000 > self.config.target_latency_ms:
                    self.logger.warning(f"Streaming latency exceeded target: {result.processing_time*1000:.1f}ms")
                
            except Exception as e:
                self.logger.error(f"Streaming processing failed for image {processed_count}: {e}")
                # Yield None to indicate failure but continue processing
                yield None
                processed_count += 1
        
        self.logger.info(f"Streaming processing completed: {processed_count} images processed")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and comprehensive health metrics.
        
        Returns:
            Dictionary with pipeline status information
        """
        # Component status
        components_status = {
            "depth_extractor": self.depth_extractor is not None,
            "synthetic_generator": self.synthetic_generator is not None,
            "wave_analyzer": self.wave_analyzer is not None
        }
        
        # Performance statistics
        performance_stats = {}
        if self._processing_history:
            processing_times = [h['processing_time'] for h in self._processing_history]
            confidence_scores = [h['confidence'] for h in self._processing_history]
            
            performance_stats = {
                'total_processed': len(self._processing_history),
                'average_processing_time': np.mean(processing_times),
                'average_confidence': np.mean(confidence_scores),
                'real_time_capable': np.mean(processing_times) * 1000 <= self.config.target_latency_ms,
                'error_rate': self._error_count / max(1, len(self._processing_history) + self._error_count)
            }
        
        # Hardware status
        hardware_status = {}
        if self.wave_analyzer:
            hardware_status = self.wave_analyzer.get_hardware_info()
        
        # Component-specific performance
        component_performance = {}
        if self.depth_extractor and hasattr(self.depth_extractor, 'get_performance_stats'):
            component_performance['depth_extractor'] = self.depth_extractor.get_performance_stats()
        
        if self.wave_analyzer and hasattr(self.wave_analyzer, 'get_performance_stats'):
            component_performance['wave_analyzer'] = self.wave_analyzer.get_performance_stats()
        
        return {
            "components_initialized": all(components_status.values()),
            "component_status": components_status,
            "config": asdict(self.config),
            "performance_statistics": performance_stats,
            "hardware_status": hardware_status,
            "component_performance": component_performance,
            "error_count": self._error_count,
            "last_error_time": self._last_error_time,
            "processing_history_size": len(self._processing_history)
        }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report.
        
        Returns:
            Dictionary with system health analysis
        """
        status = self.get_pipeline_status()
        
        # Health assessment
        health_score = 1.0
        health_issues = []
        recommendations = []
        
        # Check component health
        if not status["components_initialized"]:
            health_score *= 0.5
            health_issues.append("Not all components initialized")
            recommendations.append("Restart pipeline or check component initialization")
        
        # Check performance health
        if status.get("performance_statistics", {}).get("error_rate", 0) > 0.1:
            health_score *= 0.8
            health_issues.append("High error rate detected")
            recommendations.append("Check input data quality and system resources")
        
        if not status.get("performance_statistics", {}).get("real_time_capable", True):
            health_score *= 0.9
            health_issues.append("Not meeting real-time performance requirements")
            recommendations.append("Consider optimizing configuration or upgrading hardware")
        
        # Check recent errors
        if self._error_count > 0 and self._last_error_time:
            time_since_error = time.time() - self._last_error_time
            if time_since_error < 300:  # 5 minutes
                health_score *= 0.7
                health_issues.append("Recent errors detected")
                recommendations.append("Monitor error logs and check system stability")
        
        return {
            "health_score": health_score,
            "health_status": "HEALTHY" if health_score > 0.8 else "DEGRADED" if health_score > 0.5 else "UNHEALTHY",
            "health_issues": health_issues,
            "recommendations": recommendations,
            "detailed_status": status,
            "timestamp": time.time()
        }
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state and reinitialize components."""
        self.logger.info("Resetting pipeline...")
        
        # Clear history
        self._processing_history.clear()
        self._error_count = 0
        self._last_error_time = None
        
        # Reinitialize components
        self._initialize_components()
        
        self.logger.info("Pipeline reset completed")
    
    def _save_intermediate_results(self, results: PipelineResults, original_image: np.ndarray) -> None:
        """Save intermediate results to configured output directory.
        
        Args:
            results: Pipeline results to save
            original_image: Original input image
        """
        if not self.config.output_directory:
            return
        
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        
        try:
            # Save results JSON
            results_file = output_dir / f"results_{timestamp}.json"
            results.save_to_file(str(results_file))
            
            # Save depth maps if available
            if results.depth_map is not None:
                depth_file = output_dir / f"depth_map_{timestamp}.npy"
                np.save(depth_file, results.depth_map.data)
            
            if results.enhanced_depth_map is not None:
                enhanced_depth_file = output_dir / f"enhanced_depth_{timestamp}.npy"
                np.save(enhanced_depth_file, results.enhanced_depth_map.data)
            
            # Save original image
            original_file = output_dir / f"original_{timestamp}.npy"
            np.save(original_file, original_image)
            
            self.logger.debug(f"Intermediate results saved to {output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {e}")
    
    def _update_processing_history(self, results: PipelineResults) -> None:
        """Update processing history with latest results.
        
        Args:
            results: Latest pipeline results
        """
        history_entry = {
            'timestamp': time.time(),
            'processing_time': results.processing_time,
            'confidence': results.pipeline_confidence,
            'warnings_count': len(results.warnings),
            'depth_quality': results.depth_quality.overall_score if results.depth_quality else 0.0
        }
        
        self._processing_history.append(history_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self._processing_history) > 1000:
            self._processing_history = self._processing_history[-1000:]
    
    def configure_synthetic_generation(self, enable: bool = True) -> bool:
        """Configure synthetic data generation capability.
        
        Args:
            enable: Whether to enable synthetic generation
            
        Returns:
            True if successfully configured, False otherwise
        """
        if enable and self.synthetic_generator is None:
            try:
                self.synthetic_generator = FLUXControlNetGenerator()
                self.logger.info("Synthetic generator enabled")
                return True
            except Exception as e:
                self.logger.error(f"Failed to enable synthetic generator: {e}")
                return False
        elif not enable:
            self.synthetic_generator = None
            self.logger.info("Synthetic generator disabled")
            return True
        
        return self.synthetic_generator is not None
    
    def generate_synthetic_training_data(
        self, 
        target_size: int, 
        weather_conditions: Optional[List[WeatherConditions]] = None
    ) -> Optional[Any]:
        """Generate synthetic training data using the pipeline's synthetic generator.
        
        Args:
            target_size: Number of synthetic images to generate
            weather_conditions: Optional weather conditions for generation
            
        Returns:
            LabeledDataset if successful, None otherwise
        """
        if self.synthetic_generator is None:
            self.logger.error("Synthetic generator not available")
            return None
        
        try:
            self.logger.info(f"Generating {target_size} synthetic training images")
            dataset = self.synthetic_generator.create_balanced_dataset(target_size)
            self.logger.info(f"Successfully generated {len(dataset.images)} synthetic images")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to generate synthetic training data: {e}")
            return None