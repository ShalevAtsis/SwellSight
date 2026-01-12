"""
Progress Tracking and Reporting Utilities for SwellSight Pipeline
Provides consistent progress bars, status reporting, and performance feedback
"""

import time
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available, using basic progress tracking")

class ProgressTracker:
    """Handles progress tracking and reporting for pipeline operations"""
    
    def __init__(self):
        self.start_time = None
        self.stage_metrics = {}
    
    def create_progress_bar(self, total_items: int, description: str = "Processing",
                          unit: str = "items", show_memory: bool = False) -> Union['tqdm', 'BasicProgressBar']:
        """
        Create a progress bar for tracking operations
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
            unit: Unit of measurement for progress
            show_memory: Whether to show memory usage in progress bar
            
        Returns:
            Progress bar object
        """
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=total_items,
                desc=description,
                unit=unit,
                ncols=120 if show_memory else 100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}'
            )
            pbar.show_memory = show_memory
            return pbar
        else:
            return BasicProgressBar(total_items, description, unit, show_memory)
    
    def update_progress(self, progress_bar: Any, increment: int = 1, 
                       additional_info: Optional[str] = None) -> None:
        """
        Update progress bar with additional information
        
        Args:
            progress_bar: Progress bar object
            increment: Number of items to increment
            additional_info: Additional information to display
        """
        try:
            if hasattr(progress_bar, 'update'):
                progress_bar.update(increment)
                
                # Handle memory display if enabled
                if hasattr(progress_bar, 'show_memory') and progress_bar.show_memory:
                    try:
                        import psutil
                        memory_percent = psutil.virtual_memory().percent
                        memory_info = f"Mem: {memory_percent:.1f}%"
                        
                        if additional_info:
                            combined_info = f"{additional_info}, {memory_info}"
                        else:
                            combined_info = memory_info
                            
                        if hasattr(progress_bar, 'set_postfix_str'):
                            progress_bar.set_postfix_str(combined_info)
                    except ImportError:
                        # psutil not available, just use additional_info
                        if additional_info and hasattr(progress_bar, 'set_postfix_str'):
                            progress_bar.set_postfix_str(additional_info)
                else:
                    if additional_info and hasattr(progress_bar, 'set_postfix_str'):
                        progress_bar.set_postfix_str(additional_info)
            
        except Exception as e:
            logger.warning(f"Error updating progress: {e}")
    
    def close_progress_bar(self, progress_bar: Any) -> None:
        """
        Close progress bar properly
        
        Args:
            progress_bar: Progress bar object to close
        """
        try:
            if hasattr(progress_bar, 'close'):
                progress_bar.close()
        except Exception as e:
            logger.warning(f"Error closing progress bar: {e}")
    
    def display_stage_summary(self, stage_name: str, metrics: Dict[str, Any]) -> None:
        """
        Display summary statistics for a completed stage
        
        Args:
            stage_name: Name of the completed stage
            metrics: Dictionary of metrics and statistics
        """
        try:
            print(f"\n{'='*60}")
            print(f"STAGE SUMMARY: {stage_name.upper()}")
            print(f"{'='*60}")
            
            # Display key metrics
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'time' in key.lower() or 'duration' in key.lower():
                        print(f"{key.replace('_', ' ').title()}: {value:.2f} seconds")
                    elif 'rate' in key.lower() or 'score' in key.lower():
                        print(f"{key.replace('_', ' ').title()}: {value:.3f}")
                    else:
                        print(f"{key.replace('_', ' ').title()}: {value:.2f}")
                elif isinstance(value, int):
                    print(f"{key.replace('_', ' ').title()}: {value:,}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
            
            print(f"{'='*60}\n")
            
            # Store metrics for later reference
            self.stage_metrics[stage_name] = metrics
            
        except Exception as e:
            logger.error(f"Error displaying stage summary: {e}")
    
    def show_performance_tips(self, current_performance: Dict[str, Any]) -> None:
        """
        Show performance optimization tips based on current metrics
        
        Args:
            current_performance: Dictionary with current performance metrics
        """
        try:
            tips = []
            
            # Processing speed tips
            processing_rate = current_performance.get('items_per_second', 0)
            if processing_rate < 1.0:
                tips.append("⚡ Consider reducing batch size or image resolution for faster processing")
                tips.append("🔧 Check if GPU acceleration is enabled and working properly")
            
            # Memory usage tips
            memory_usage = current_performance.get('memory_usage_percent', 0)
            if memory_usage > 80:
                tips.append("🧠 High memory usage detected - consider reducing batch size")
                tips.append("🗑️ Clear unused variables and call garbage collection")
            
            # Error rate tips
            error_rate = current_performance.get('error_rate', 0)
            if error_rate > 0.1:  # More than 10% errors
                tips.append("⚠️ High error rate - check input data quality")
                tips.append("🔍 Review error logs for common failure patterns")
            
            # Quality tips
            quality_score = current_performance.get('average_quality_score', 1.0)
            if quality_score < 0.7:
                tips.append("📊 Low quality scores - consider adjusting quality thresholds")
                tips.append("🎯 Review input data for quality issues")
            
            # Display tips if any
            if tips:
                print(f"\n{'🚀 PERFORMANCE TIPS':^60}")
                print("─" * 60)
                for tip in tips[:5]:  # Show top 5 tips
                    print(f"  {tip}")
                print("─" * 60)
            
        except Exception as e:
            logger.error(f"Error showing performance tips: {e}")
    
    @contextmanager
    def track_stage(self, stage_name: str):
        """
        Context manager for tracking stage execution time and metrics
        
        Args:
            stage_name: Name of the stage being tracked
        """
        start_time = time.time()
        stage_metrics = {
            'stage_name': stage_name,
            'start_time': start_time,
            'items_processed': 0,
            'errors_encountered': 0,
            'warnings_generated': 0
        }
        
        try:
            logger.info(f"Starting stage: {stage_name}")
            yield stage_metrics
            
        except Exception as e:
            stage_metrics['error'] = str(e)
            logger.error(f"Stage {stage_name} failed: {e}")
            raise
            
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            stage_metrics.update({
                'end_time': end_time,
                'duration_seconds': duration,
                'items_per_second': stage_metrics['items_processed'] / max(duration, 0.001)
            })
            
            # Calculate success rate
            total_items = stage_metrics['items_processed'] + stage_metrics['errors_encountered']
            if total_items > 0:
                stage_metrics['success_rate'] = stage_metrics['items_processed'] / total_items
            else:
                stage_metrics['success_rate'] = 1.0
            
            logger.info(f"Completed stage: {stage_name} in {duration:.2f} seconds")
            
            # Display summary
            self.display_stage_summary(stage_name, stage_metrics)
    
    def generate_pipeline_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report for the entire pipeline
        
        Returns:
            Dictionary with pipeline metrics and summary
        """
        try:
            if not self.stage_metrics:
                return {'error': 'No stage metrics available'}
            
            # Calculate overall metrics
            total_duration = sum(
                metrics.get('duration_seconds', 0) 
                for metrics in self.stage_metrics.values()
            )
            
            total_items = sum(
                metrics.get('items_processed', 0) 
                for metrics in self.stage_metrics.values()
            )
            
            total_errors = sum(
                metrics.get('errors_encountered', 0) 
                for metrics in self.stage_metrics.values()
            )
            
            # Generate report
            report = {
                'pipeline_summary': {
                    'total_stages': len(self.stage_metrics),
                    'total_duration_seconds': total_duration,
                    'total_items_processed': total_items,
                    'total_errors': total_errors,
                    'overall_success_rate': (total_items / max(total_items + total_errors, 1)),
                    'average_processing_rate': total_items / max(total_duration, 0.001)
                },
                'stage_details': self.stage_metrics,
                'recommendations': self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating pipeline report: {e}")
            return {'error': f'Report generation failed: {e}'}
    
    def generate_quality_summary_report(self, quality_data: Dict[str, Any], 
                                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive quality summary report
        
        Args:
            quality_data: Dictionary containing quality metrics and statistics
            output_path: Optional path to save the report
            
        Returns:
            Dictionary with formatted quality report
        """
        try:
            import datetime
            
            # Create comprehensive quality report
            report = {
                'report_metadata': {
                    'generated_at': datetime.datetime.now().isoformat(),
                    'report_type': 'Quality Summary Report',
                    'pipeline_stage': 'Data Preprocessing'
                },
                'dataset_overview': {
                    'total_images_processed': quality_data.get('total_images_processed', 0),
                    'valid_images': quality_data.get('valid_images_count', 0),
                    'invalid_images': quality_data.get('invalid_images_count', 0),
                    'success_rate_percent': quality_data.get('success_rate', 0) * 100,
                    'total_size_mb': quality_data.get('total_size_mb', 0)
                },
                'quality_metrics': {
                    'average_quality_score': quality_data.get('average_quality_score', 0),
                    'quality_score_std': quality_data.get('quality_score_std', 0),
                    'average_brightness': quality_data.get('average_brightness', 0),
                    'average_contrast': quality_data.get('average_contrast', 0),
                    'average_sharpness': quality_data.get('average_sharpness', 0)
                },
                'distribution_analysis': {
                    'resolution_distribution': quality_data.get('resolution_distribution', {}),
                    'format_distribution': quality_data.get('format_distribution', {}),
                    'color_mode_distribution': quality_data.get('color_mode_distribution', {})
                },
                'processing_performance': {
                    'processing_time_seconds': quality_data.get('processing_time_seconds', 0),
                    'images_per_second': quality_data.get('images_per_second', 0),
                    'batch_size_used': quality_data.get('batch_size_used', 0)
                },
                'recommendations': quality_data.get('recommendations', {}),
                'issues_found': quality_data.get('processing_errors', [])
            }
            
            # Add quality assessment
            avg_quality = quality_data.get('average_quality_score', 0)
            quality_threshold = quality_data.get('quality_threshold', 0.7)
            
            report['quality_assessment'] = {
                'overall_quality_rating': self._get_quality_rating(avg_quality),
                'meets_threshold': avg_quality >= quality_threshold,
                'quality_threshold': quality_threshold,
                'ready_for_next_stage': quality_data.get('dataset_ready_for_next_stage', False)
            }
            
            # Save report if path provided
            if output_path:
                import json
                from pathlib import Path
                
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Quality summary report saved to: {output_file}")
            
            # Display summary
            self._display_quality_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality summary report: {e}")
            return {'error': f'Report generation failed: {e}'}
    
    def _get_quality_rating(self, quality_score: float) -> str:
        """Get quality rating based on score"""
        if quality_score >= 0.9:
            return "Excellent"
        elif quality_score >= 0.8:
            return "Good"
        elif quality_score >= 0.7:
            return "Acceptable"
        elif quality_score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def _display_quality_summary(self, report: Dict[str, Any]) -> None:
        """Display formatted quality summary"""
        try:
            print(f"\n{'='*80}")
            print(f"{'QUALITY SUMMARY REPORT':^80}")
            print(f"{'='*80}")
            
            # Dataset overview
            overview = report['dataset_overview']
            print(f"\n📊 DATASET OVERVIEW:")
            print(f"   Total Images Processed: {overview['total_images_processed']:,}")
            print(f"   Valid Images: {overview['valid_images']:,} ({overview['success_rate_percent']:.1f}%)")
            print(f"   Invalid Images: {overview['invalid_images']:,}")
            print(f"   Total Size: {overview['total_size_mb']:.1f} MB")
            
            # Quality metrics
            metrics = report['quality_metrics']
            assessment = report['quality_assessment']
            print(f"\n🎯 QUALITY METRICS:")
            print(f"   Overall Quality Rating: {assessment['overall_quality_rating']}")
            print(f"   Average Quality Score: {metrics['average_quality_score']:.3f}")
            print(f"   Average Brightness: {metrics['average_brightness']:.1f}")
            print(f"   Average Contrast: {metrics['average_contrast']:.3f}")
            print(f"   Average Sharpness: {metrics['average_sharpness']:.3f}")
            
            # Performance
            performance = report['processing_performance']
            print(f"\n⚡ PROCESSING PERFORMANCE:")
            print(f"   Processing Time: {performance['processing_time_seconds']:.1f} seconds")
            print(f"   Processing Rate: {performance['images_per_second']:.1f} images/second")
            print(f"   Batch Size Used: {performance['batch_size_used']}")
            
            # Status
            print(f"\n🚀 PIPELINE STATUS:")
            if assessment['ready_for_next_stage']:
                print(f"   ✅ Dataset is ready for next pipeline stage")
            else:
                print(f"   ⚠️  Dataset may need improvements before proceeding")
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"Error displaying quality summary: {e}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on pipeline performance"""
        recommendations = []
        
        try:
            if not self.stage_metrics:
                return recommendations
            
            # Analyze stage performance
            slowest_stage = max(
                self.stage_metrics.items(),
                key=lambda x: x[1].get('duration_seconds', 0)
            )
            
            fastest_stage = min(
                self.stage_metrics.items(),
                key=lambda x: x[1].get('duration_seconds', float('inf'))
            )
            
            recommendations.append(
                f"Slowest stage: {slowest_stage[0]} "
                f"({slowest_stage[1].get('duration_seconds', 0):.1f}s)"
            )
            
            # Error analysis
            error_stages = [
                name for name, metrics in self.stage_metrics.items()
                if metrics.get('errors_encountered', 0) > 0
            ]
            
            if error_stages:
                recommendations.append(f"Stages with errors: {', '.join(error_stages)}")
            
            # Performance recommendations
            avg_rate = sum(
                metrics.get('items_per_second', 0) 
                for metrics in self.stage_metrics.values()
            ) / len(self.stage_metrics)
            
            if avg_rate < 1.0:
                recommendations.append("Consider optimizing processing speed")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations"]


class BasicProgressBar:
    """Basic progress bar implementation when tqdm is not available"""
    
    def __init__(self, total: int, description: str = "Processing", unit: str = "items", show_memory: bool = False):
        self.total = total
        self.description = description
        self.unit = unit
        self.show_memory = show_memory
        self.current = 0
        self.start_time = time.time()
        self.additional_info = ""
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        self._display_progress()
    
    def _display_progress(self):
        """Display current progress"""
        if self.total > 0:
            percent = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            rate = self.current / max(elapsed, 0.001)
            
            progress_str = f"\r{self.description}: {self.current}/{self.total} " \
                          f"({percent:.1f}%) [{rate:.1f} {self.unit}/s]"
            
            if self.additional_info:
                progress_str += f" | {self.additional_info}"
            
            print(progress_str, end="", flush=True)
            
            if self.current >= self.total:
                print()  # New line when complete
    
    def set_postfix_str(self, postfix: str):
        """Set additional information"""
        self.additional_info = postfix
        self._display_progress()
    
    def close(self):
        """Close progress bar"""
        if self.current < self.total:
            print()  # Ensure new line


# Convenience functions for direct use in notebooks
def create_progress_bar(total_items: int, description: str = "Processing", show_memory: bool = False) -> Any:
    """Create a progress bar for tracking operations"""
    tracker = ProgressTracker()
    return tracker.create_progress_bar(total_items, description, show_memory=show_memory)

def display_stage_summary(stage_name: str, metrics: Dict[str, Any]) -> None:
    """Display summary statistics for a completed stage"""
    tracker = ProgressTracker()
    tracker.display_stage_summary(stage_name, metrics)