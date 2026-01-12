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
                          unit: str = "items") -> Union['tqdm', 'BasicProgressBar']:
        """
        Create a progress bar for tracking operations
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
            unit: Unit of measurement for progress
            
        Returns:
            Progress bar object
        """
        if TQDM_AVAILABLE:
            return tqdm(
                total=total_items,
                desc=description,
                unit=unit,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            return BasicProgressBar(total_items, description, unit)
    
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
                
                if additional_info and hasattr(progress_bar, 'set_postfix_str'):
                    progress_bar.set_postfix_str(additional_info)
            
        except Exception as e:
            logger.warning(f"Error updating progress: {e}")
    
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
    
    def __init__(self, total: int, description: str = "Processing", unit: str = "items"):
        self.total = total
        self.description = description
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
    
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
            
            print(f"\r{self.description}: {self.current}/{self.total} "
                  f"({percent:.1f}%) [{rate:.1f} {self.unit}/s]", end="", flush=True)
            
            if self.current >= self.total:
                print()  # New line when complete
    
    def set_postfix_str(self, postfix: str):
        """Set additional information"""
        # Basic implementation - just print the info
        pass
    
    def close(self):
        """Close progress bar"""
        if self.current < self.total:
            print()  # Ensure new line


# Convenience functions for direct use in notebooks
def create_progress_bar(total_items: int, description: str = "Processing") -> Any:
    """Create a progress bar for tracking operations"""
    tracker = ProgressTracker()
    return tracker.create_progress_bar(total_items, description)

def display_stage_summary(stage_name: str, metrics: Dict[str, Any]) -> None:
    """Display summary statistics for a completed stage"""
    tracker = ProgressTracker()
    tracker.display_stage_summary(stage_name, metrics)