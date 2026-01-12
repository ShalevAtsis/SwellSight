"""
Enhanced Batch Processor for FLUX ControlNet Generation
Integrates advanced memory management, quality validation, and data distribution comparison
"""

import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional
import time
import json
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

from flux_memory_manager import FLUXMemoryManager, QualityValidator, DataDistributionComparator
from progress_tracker import ProgressTracker
from error_handler import ErrorHandler

@dataclass
class BatchProcessingResult:
    """Comprehensive result structure for batch processing"""
    total_processed: int
    successful: int
    failed: int
    success_rate: float
    processing_time: float
    memory_usage: Dict[str, Any]
    quality_summary: Dict[str, Any]
    distribution_comparison: Optional[Dict[str, Any]]
    generated_files: List[str]
    error_log: List[Dict[str, Any]]

class EnhancedBatchProcessor:
    """Enhanced batch processor with advanced memory management and quality validation"""
    
    def __init__(self, generator, config: Dict[str, Any]):
        self.generator = generator
        self.config = config
        
        # Initialize advanced components
        self.memory_manager = FLUXMemoryManager(config)
        self.quality_validator = QualityValidator(config)
        self.distribution_comparator = DataDistributionComparator(config)
        self.error_handler = ErrorHandler()
        
        # Processing state
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'quality_scores': [],
            'processing_times': [],
            'memory_profiles': [],
            'error_log': []
        }
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def calculate_dynamic_batch_size(self, image_resolution: Tuple[int, int] = (1024, 1024)) -> int:
        """Calculate optimal batch size with advanced memory analysis"""
        
        # Get memory health check
        memory_health = self.memory_manager.check_memory_health()
        
        if memory_health['status'] == 'critical':
            self.logger.warning("Critical memory usage detected - using minimal batch size")
            return 1
        
        # Calculate base batch size
        base_batch_size = self.memory_manager.calculate_optimal_batch_size(image_resolution)
        
        # Adjust based on memory health
        if memory_health['status'] == 'warning':
            adjusted_batch_size = max(1, base_batch_size // 2)
            self.logger.info(f"Memory warning - reducing batch size from {base_batch_size} to {adjusted_batch_size}")
            return adjusted_batch_size
        
        return base_batch_size
    
    def process_single_image(self, depth_map: np.ndarray, params: Dict[str, Any], 
                           image_index: int) -> Tuple[bool, Dict[str, Any]]:
        """Process a single image with comprehensive monitoring"""
        
        # Start memory monitoring
        memory_monitor = self.memory_manager.monitor_memory_during_generation(f"image_{image_index}")
        
        try:
            # Validate input depth map
            if depth_map.size == 0 or np.all(depth_map == 0):
                raise ValueError("Invalid depth map: empty or all zeros")
            
            # Generate synthetic image
            start_time = time.time()
            result = self.generator.generate_with_memory_management(depth_map, params)
            generation_time = time.time() - start_time
            
            # Validate generated image quality
            quality_metrics = self.quality_validator.validate_image_quality(
                result.synthetic_image, depth_map
            )
            
            # Finalize memory monitoring
            memory_stats = self.memory_manager.finalize_memory_monitoring(memory_monitor)
            
            # Prepare success result
            success_result = {
                'image_index': image_index,
                'generation_result': result,
                'quality_metrics': quality_metrics,
                'memory_stats': memory_stats,
                'processing_time': generation_time,
                'status': 'success'
            }
            
            return True, success_result
            
        except torch.cuda.OutOfMemoryError as e:
            # Handle GPU memory error with fallback
            self.logger.warning(f"GPU memory error for image {image_index}: {e}")
            
            try:
                # Cleanup and retry with reduced parameters
                self.memory_manager.cleanup_memory(aggressive=True)
                
                # Reduce parameters
                reduced_params = params.copy()
                reduced_params['height'] = min(params.get('height', 1024), 768)
                reduced_params['width'] = min(params.get('width', 1024), 768)
                reduced_params['num_inference_steps'] = min(params.get('num_inference_steps', 28), 20)
                
                # Retry generation
                result = self.generator.generate_with_memory_management(depth_map, reduced_params)
                quality_metrics = self.quality_validator.validate_image_quality(
                    result.synthetic_image, depth_map
                )
                
                memory_stats = self.memory_manager.finalize_memory_monitoring(memory_monitor)
                
                success_result = {
                    'image_index': image_index,
                    'generation_result': result,
                    'quality_metrics': quality_metrics,
                    'memory_stats': memory_stats,
                    'processing_time': time.time() - start_time,
                    'status': 'success_with_fallback',
                    'fallback_params': reduced_params
                }
                
                return True, success_result
                
            except Exception as retry_error:
                error_result = {
                    'image_index': image_index,
                    'error_type': 'gpu_memory_error_with_failed_retry',
                    'original_error': str(e),
                    'retry_error': str(retry_error),
                    'status': 'failed'
                }
                return False, error_result
        
        except Exception as e:
            # Handle other errors
            self.logger.error(f"Generation failed for image {image_index}: {e}")
            
            error_result = {
                'image_index': image_index,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'status': 'failed'
            }
            
            return False, error_result
    
    def save_generated_image(self, result: Dict[str, Any], output_dir: Path) -> str:
        """Save generated image with metadata"""
        image_index = result['image_index']
        generation_result = result['generation_result']
        
        # Generate filename
        filename = f"flux_synthetic_{image_index:04d}.png"
        image_path = output_dir / filename
        
        # Save image
        generation_result.synthetic_image.save(image_path, "PNG")
        
        # Save metadata
        metadata = {
            'filename': filename,
            'image_index': image_index,
            'generation_params': generation_result.generation_params,
            'quality_metrics': result['quality_metrics'],
            'memory_stats': result['memory_stats'],
            'processing_time': result['processing_time'],
            'model_info': generation_result.model_info,
            'status': result['status']
        }
        
        metadata_filename = f"flux_synthetic_{image_index:04d}_metadata.json"
        metadata_path = output_dir / "metadata" / metadata_filename
        metadata_path.parent.mkdir(exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return filename
    
    def update_processing_stats(self, success: bool, result: Dict[str, Any]):
        """Thread-safe update of processing statistics"""
        with self.stats_lock:
            self.processing_stats['total_processed'] += 1
            
            if success:
                self.processing_stats['successful'] += 1
                
                # Extract quality and timing info
                if 'quality_metrics' in result:
                    overall_score = result['quality_metrics'].get('overall_score', 0.0)
                    self.processing_stats['quality_scores'].append(overall_score)
                
                if 'processing_time' in result:
                    self.processing_stats['processing_times'].append(result['processing_time'])
                
                if 'memory_stats' in result:
                    self.processing_stats['memory_profiles'].append(result['memory_stats'])
            else:
                self.processing_stats['failed'] += 1
                self.processing_stats['error_log'].append(result)
    
    def process_batch_with_quality_validation(self, depth_maps: List[np.ndarray], 
                                            params_list: List[Dict[str, Any]],
                                            output_dir: Path,
                                            reference_images: List[Image.Image] = None) -> BatchProcessingResult:
        """Process batch with comprehensive quality validation and monitoring"""
        
        # Ensure output directories exist
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata").mkdir(exist_ok=True)
        
        # Calculate optimal batch size
        sample_resolution = (params_list[0].get('width', 1024), params_list[0].get('height', 1024))
        batch_size = self.calculate_dynamic_batch_size(sample_resolution)
        
        total_items = min(len(depth_maps), len(params_list))
        generated_files = []
        
        self.logger.info(f"Starting enhanced batch processing: {total_items} items, batch size: {batch_size}")
        
        # Initialize progress tracking
        progress = ProgressTracker(total_items, "FLUX Generation with Quality Validation")
        
        start_time = time.time()
        
        try:
            # Process in batches
            for batch_start in range(0, total_items, batch_size):
                batch_end = min(batch_start + batch_size, total_items)
                batch_depth_maps = depth_maps[batch_start:batch_end]
                batch_params = params_list[batch_start:batch_end]
                
                self.logger.info(f"Processing batch {batch_start//batch_size + 1}: items {batch_start+1}-{batch_end}")
                
                # Check memory health before batch
                memory_health = self.memory_manager.check_memory_health()
                if memory_health['status'] == 'critical':
                    self.logger.warning("Critical memory usage - performing cleanup")
                    self.memory_manager.cleanup_memory(aggressive=True)
                
                # Process batch items
                batch_futures = []\
                
                # Use ThreadPoolExecutor for I/O operations (saving files)\n",
                with ThreadPoolExecutor(max_workers=2) as executor:\n",
                    for i, (depth_map, params) in enumerate(zip(batch_depth_maps, batch_params)):\n",
                        image_index = batch_start + i\n",
                        \n",
                        # Process single image\n",
                        success, result = self.process_single_image(depth_map, params, image_index)\n",
                        \n",
                        # Update statistics\n",
                        self.update_processing_stats(success, result)\n",
                        \n",
                        if success:\n",
                            # Submit file saving to thread pool\n",
                            future = executor.submit(self.save_generated_image, result, output_dir)\n",
                            batch_futures.append(future)\n",
                        \n",
                        # Update progress\n",
                        progress.update(1)\n",
                        \n",
                        # Memory cleanup between images if needed\n",
                        if (i + 1) % 3 == 0:  # Cleanup every 3 images\n",
                            self.memory_manager.cleanup_memory()\n",
                    \n",
                    # Wait for all file operations to complete\n",
                    for future in as_completed(batch_futures):\n",
                        try:\n",
                            filename = future.result()\n",
                            generated_files.append(filename)\n",
                        except Exception as e:\n",
                            self.logger.error(f\"File saving failed: {e}\")\n",
                \n",
                # Batch-level memory cleanup\n",
                self.memory_manager.cleanup_memory()\n",
                \n",
                # Log batch completion\n",
                current_memory = self.memory_manager.get_current_memory_profile()\n",
                self.logger.info(f\"Batch completed. GPU Memory: {current_memory.gpu_allocated_gb:.2f} GB\")\n",
        \n",
        finally:\n",
            progress.close()\n",
        \n",
        total_time = time.time() - start_time\n",
        \n",
        # Calculate final statistics\n",
        with self.stats_lock:\n",
            success_rate = self.processing_stats['successful'] / max(1, self.processing_stats['total_processed'])\n",
            \n",
            quality_summary = {\n",
                'mean_quality': np.mean(self.processing_stats['quality_scores']) if self.processing_stats['quality_scores'] else 0.0,\n",
                'std_quality': np.std(self.processing_stats['quality_scores']) if self.processing_stats['quality_scores'] else 0.0,\n",
                'min_quality': np.min(self.processing_stats['quality_scores']) if self.processing_stats['quality_scores'] else 0.0,\n",
                'max_quality': np.max(self.processing_stats['quality_scores']) if self.processing_stats['quality_scores'] else 0.0,\n",
                'acceptable_count': sum(1 for score in self.processing_stats['quality_scores'] if score >= 0.4)\n",
            }\n",
            \n",
            # Memory usage summary\n",
            memory_summary = {\n",
                'peak_gpu_usage': max([p.get('peak_gpu_usage', 0) for p in self.processing_stats['memory_profiles']], default=0),\n",
                'average_gpu_usage': np.mean([p.get('peak_gpu_usage', 0) for p in self.processing_stats['memory_profiles']]) if self.processing_stats['memory_profiles'] else 0,\n",
                'memory_profiles_count': len(self.processing_stats['memory_profiles'])\n",
            }\n",
        \n",
        # Perform distribution comparison if reference images provided\n",
        distribution_comparison = None\n",
        if reference_images and generated_files:\n",
            try:\n",
                # Load generated images for comparison\n",
                generated_images = []\n",
                for filename in generated_files[:20]:  # Limit for performance\n",
                    try:\n",
                        img_path = output_dir / filename\n",
                        if img_path.exists():\n",
                            img = Image.open(img_path)\n",
                            generated_images.append(img)\n",
                    except Exception as e:\n",
                        self.logger.warning(f\"Could not load generated image {filename}: {e}\")\n",
                \n",
                if generated_images:\n",
                    self.logger.info(f\"Comparing distributions: {len(generated_images)} synthetic vs {len(reference_images)} real\")\n",
                    distribution_comparison = self.distribution_comparator.compare_distributions(\n",
                        generated_images, reference_images\n",
                    )\n",
            except Exception as e:\n",
                self.logger.error(f\"Distribution comparison failed: {e}\")\n",
        \n",
        # Create comprehensive result\n",
        result = BatchProcessingResult(\n",
            total_processed=self.processing_stats['total_processed'],\n",
            successful=self.processing_stats['successful'],\n",
            failed=self.processing_stats['failed'],\n",
            success_rate=success_rate,\n",
            processing_time=total_time,\n",
            memory_usage=memory_summary,\n",
            quality_summary=quality_summary,\n",
            distribution_comparison=distribution_comparison,\n",
            generated_files=generated_files,\n",
            error_log=self.processing_stats['error_log']\n",
        )\n",
        \n",
        # Save comprehensive results\n",
        self.save_batch_results(result, output_dir)\n",
        \n",
        return result\n",
    \n",
    def save_batch_results(self, result: BatchProcessingResult, output_dir: Path):\n",
        \"\"\"Save comprehensive batch processing results\"\"\"\n",
        results_file = output_dir / \"batch_processing_results.json\"\n",
        \n",
        # Convert result to dictionary\n",
        result_dict = asdict(result)\n",
        result_dict['timestamp'] = time.time()\n",
        result_dict['output_directory'] = str(output_dir)\n",
        \n",
        with open(results_file, 'w') as f:\n",
            json.dump(result_dict, f, indent=2, default=str)\n",
        \n",
        self.logger.info(f\"Batch results saved to {results_file}\")\n",
    \n",
    def generate_processing_report(self, result: BatchProcessingResult, output_dir: Path) -> str:\n",
        \"\"\"Generate comprehensive processing report\"\"\"\n",
        report_lines = [\n",
            \"# FLUX ControlNet Batch Processing Report\",\n",
            f\"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\",\n",
            \"\",\n",
            \"## Processing Summary\",\n",
            f\"- Total Images Processed: {result.total_processed}\",\n",
            f\"- Successful Generations: {result.successful}\",\n",
            f\"- Failed Generations: {result.failed}\",\n",
            f\"- Success Rate: {result.success_rate:.1%}\",\n",
            f\"- Total Processing Time: {result.processing_time:.1f} seconds\",\n",
            f\"- Average Time per Image: {result.processing_time / max(1, result.successful):.1f} seconds\",\n",
            \"\",\n",
            \"## Quality Analysis\",\n",
            f\"- Mean Quality Score: {result.quality_summary['mean_quality']:.3f}\",\n",
            f\"- Quality Standard Deviation: {result.quality_summary['std_quality']:.3f}\",\n",
            f\"- Quality Range: [{result.quality_summary['min_quality']:.3f}, {result.quality_summary['max_quality']:.3f}]\",\n",
            f\"- Acceptable Quality Images: {result.quality_summary['acceptable_count']}/{result.successful}\",\n",
            \"\",\n",
            \"## Memory Usage\",\n",
            f\"- Peak GPU Memory: {result.memory_usage['peak_gpu_usage']:.2f} GB\",\n",
            f\"- Average GPU Memory: {result.memory_usage['average_gpu_usage']:.2f} GB\",\n",
            \"\"\n",
        ]\n",
        \n",
        # Add distribution comparison if available\n",
        if result.distribution_comparison:\n",
            comp = result.distribution_comparison\n",
            report_lines.extend([\n",
                \"## Distribution Comparison (Synthetic vs Real)\",\n",
                f\"- Overall Similarity: {comp['overall_similarity']:.1%}\",\n",
                f\"- Assessment: {comp['assessment']['level'].title()}\",\n",
                f\"- {comp['assessment']['message']}\",\n",
                f\"- Synthetic Images: {comp['synthetic_count']}\",\n",
                f\"- Real Images: {comp['real_count']}\",\n",
                \"\"\n",
            ])\n",
            \n",
            # Add feature-specific comparisons\n",
            report_lines.append(\"### Feature Comparisons\")\n",
            for feature, comparison in comp['feature_comparisons'].items():\n",
                report_lines.extend([\n",
                    f\"#### {feature.replace('_', ' ').title()}\",\n",
                    f\"- Similarity Score: {comparison['similarity_score']:.1%}\",\n",
                    f\"- Synthetic Mean: {comparison['synthetic_stats']['mean']:.3f}\",\n",
                    f\"- Real Mean: {comparison['real_stats']['mean']:.3f}\",\n",
                    f\"- Mean Difference: {comparison['mean_difference']:.3f}\",\n",
                    \"\"\n",
                ])\n",
        \n",
        # Add error analysis if there were failures\n",
        if result.failed > 0:\n",
            report_lines.extend([\n",
                \"## Error Analysis\",\n",
                f\"- Total Errors: {result.failed}\",\n",
                \"\"\n",
            ])\n",
            \n",
            # Group errors by type\n",
            error_types = {}\n",
            for error in result.error_log:\n",
                error_type = error.get('error_type', 'unknown')\n",
                error_types[error_type] = error_types.get(error_type, 0) + 1\n",
            \n",
            for error_type, count in error_types.items():\n",
                report_lines.append(f\"- {error_type}: {count} occurrences\")\n",
        \n",
        report_text = \"\\n\".join(report_lines)\n",
        \n",
        # Save report\n",
        report_file = output_dir / \"processing_report.md\"\n",
        with open(report_file, 'w') as f:\n",
            f.write(report_text)\n",
        \n",
        self.logger.info(f\"Processing report saved to {report_file}\")\n",
        return report_text"