"""
Error Handling Utilities for SwellSight Pipeline
Provides retry logic, fallback mechanisms, and user-friendly error messages
"""

import time
import functools
import torch
import logging
from typing import Callable, Any, Optional, Dict, List, Union
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Handles error recovery and user guidance for pipeline operations"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
    
    def retry_with_backoff(self, func: Callable, max_retries: int = 3, 
                          backoff_factor: float = 2.0, 
                          initial_delay: float = 1.0) -> Any:
        """
        Retry function with exponential backoff
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            backoff_factor: Multiplier for delay between retries
            initial_delay: Initial delay in seconds
            
        Returns:
            Function result if successful
            
        Raises:
            Exception: Last exception if all retries fail
        """
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                result = func()
                if attempt > 0:
                    logger.info(f"Operation succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        # If we get here, all retries failed
        raise last_exception
    
    def handle_gpu_memory_error(self, operation: str, 
                               fallback_func: Optional[Callable] = None) -> Any:
        """
        Handle GPU memory errors with CPU fallback
        
        Args:
            operation: Description of the operation for logging
            fallback_func: Function to call for CPU fallback
            
        Returns:
            Result from fallback function if provided
        """
        try:
            if self.gpu_available:
                # Clear GPU cache
                torch.cuda.empty_cache()
                logger.warning(f"GPU memory error in {operation}. Cleared GPU cache.")
                
                # Provide memory usage info
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    logger.info(f"GPU memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
            
            # Execute fallback if provided
            if fallback_func:
                logger.info(f"Falling back to CPU processing for {operation}")
                return fallback_func()
            else:
                logger.error(f"No fallback available for {operation}")
                return None
                
        except Exception as e:
            logger.error(f"Error in GPU fallback handling: {e}")
            return None
    
    def handle_file_operation_error(self, file_path: Union[str, Path], 
                                  operation: str, error: Exception) -> Dict[str, Any]:
        """
        Handle file operation errors with helpful guidance
        
        Args:
            file_path: Path to the file
            operation: Description of the operation
            error: The exception that occurred
            
        Returns:
            Dictionary with error info and suggestions
        """
        try:
            file_path = Path(file_path)
            
            error_info = {
                'operation': operation,
                'file_path': str(file_path),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'suggestions': []
            }
            
            # Provide specific suggestions based on error type
            if isinstance(error, FileNotFoundError):
                error_info['suggestions'].extend([
                    f"Check if the file exists: {file_path}",
                    f"Verify the file path is correct",
                    f"Ensure previous pipeline stages completed successfully"
                ])
                
                # Check if parent directory exists
                if not file_path.parent.exists():
                    error_info['suggestions'].append(f"Create parent directory: {file_path.parent}")
            
            elif isinstance(error, PermissionError):
                error_info['suggestions'].extend([
                    f"Check file permissions for: {file_path}",
                    f"Ensure you have read/write access to the directory",
                    f"Try running with appropriate permissions"
                ])
            
            elif isinstance(error, OSError):
                error_info['suggestions'].extend([
                    f"Check available disk space",
                    f"Verify the file is not corrupted",
                    f"Try copying the file to a different location"
                ])
            
            else:
                error_info['suggestions'].extend([
                    f"Check file format and integrity",
                    f"Verify file is not in use by another process",
                    f"Try the operation again after a short delay"
                ])
            
            return error_info
            
        except Exception as e:
            logger.error(f"Error handling file operation error: {e}")
            return {
                'operation': operation,
                'file_path': str(file_path),
                'error_type': 'Unknown',
                'error_message': 'Error handling failed',
                'suggestions': ['Contact support for assistance']
            }
    
    def save_partial_results(self, results: Dict[str, Any], 
                           stage_name: str, 
                           output_dir: str = "./outputs") -> bool:
        """
        Save partial results when errors occur
        
        Args:
            results: Partial results to save
            stage_name: Name of the pipeline stage
            output_dir: Output directory
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stage_name}_partial_results_{timestamp}.json"
            filepath = output_path / filename
            
            # Save results as JSON
            import json
            with open(filepath, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Partial results saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save partial results: {e}")
            return False
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def provide_recovery_instructions(self, error_type: str, 
                                    stage_name: str) -> List[str]:
        """
        Provide recovery instructions based on error type and stage
        
        Args:
            error_type: Type of error that occurred
            stage_name: Pipeline stage where error occurred
            
        Returns:
            List of recovery instructions
        """
        try:
            instructions = []
            
            # General instructions
            instructions.append(f"Error occurred in stage: {stage_name}")
            instructions.append("Check the error logs above for specific details")
            
            # Stage-specific instructions
            stage_instructions = {
                'setup': [
                    "Verify all dependencies are installed correctly",
                    "Check GPU availability and drivers",
                    "Ensure sufficient disk space for models and data"
                ],
                'data_preprocessing': [
                    "Check input data directory and file formats",
                    "Verify image files are not corrupted",
                    "Ensure sufficient memory for batch processing"
                ],
                'depth_extraction': [
                    "Verify depth model is downloaded correctly",
                    "Check GPU memory availability",
                    "Try reducing batch size if memory errors occur"
                ],
                'augmentation': [
                    "Check parameter ranges and validation",
                    "Verify output directory permissions",
                    "Ensure previous stages completed successfully"
                ],
                'synthetic_generation': [
                    "Verify FLUX model and ControlNet are available",
                    "Check GPU memory for large model inference",
                    "Try reducing generation batch size"
                ],
                'training': [
                    "Check training data availability and format",
                    "Verify model checkpoint directory permissions",
                    "Monitor GPU memory during training"
                ],
                'evaluation': [
                    "Ensure trained model file exists",
                    "Check test data availability",
                    "Verify output directory for results"
                ]
            }
            
            # Add stage-specific instructions
            for key, stage_instructions_list in stage_instructions.items():
                if key in stage_name.lower():
                    instructions.extend(stage_instructions_list)
                    break
            
            # Error-type specific instructions
            if 'memory' in error_type.lower():
                instructions.extend([
                    "Reduce batch size in configuration",
                    "Clear GPU cache: torch.cuda.empty_cache()",
                    "Close other GPU-using processes"
                ])
            elif 'file' in error_type.lower():
                instructions.extend([
                    "Check file paths and permissions",
                    "Verify files are not corrupted",
                    "Ensure sufficient disk space"
                ])
            elif 'network' in error_type.lower():
                instructions.extend([
                    "Check internet connection",
                    "Try downloading models manually",
                    "Use cached models if available"
                ])
            
            # Recovery actions
            instructions.extend([
                "Review partial results if saved",
                "Restart from the failed stage after fixing issues",
                "Consider reducing processing parameters if memory issues persist"
            ])
            
            return instructions
            
        except Exception as e:
            logger.error(f"Error generating recovery instructions: {e}")
            return [
                "Unable to generate specific recovery instructions",
                "Check error logs and documentation",
                "Contact support if issues persist"
            ]


# Convenience functions for direct use in notebooks
def retry_with_backoff(func: Callable, max_retries: int = 3, 
                      backoff_factor: float = 2.0) -> Any:
    """Retry function with exponential backoff"""
    handler = ErrorHandler()
    return handler.retry_with_backoff(func, max_retries, backoff_factor)

def handle_gpu_memory_error(operation: str, fallback_func: Optional[Callable] = None) -> Any:
    """Handle GPU memory errors with CPU fallback"""
    handler = ErrorHandler()
    return handler.handle_gpu_memory_error(operation, fallback_func)

def save_partial_results(results: Dict[str, Any], stage_name: str, 
                        output_dir: str = "./outputs") -> bool:
    """Save partial results when errors occur"""
    handler = ErrorHandler()
    return handler.save_partial_results(results, stage_name, output_dir)