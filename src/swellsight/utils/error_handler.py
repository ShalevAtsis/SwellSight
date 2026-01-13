"""
Comprehensive error handling and recovery system for SwellSight.

Provides retry logic, graceful degradation, and informative error messages
with recovery guidance for all system components.
"""

import time
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass
import torch
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for targeted handling."""
    INPUT_VALIDATION = "input_validation"
    MODEL_LOADING = "model_loading"
    PROCESSING = "processing"
    MEMORY = "memory"
    HARDWARE = "hardware"
    NETWORK = "network"
    FILE_IO = "file_io"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    error_type: Type[Exception]
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    operation: str
    timestamp: float
    traceback_str: str
    recovery_suggestions: List[str]
    user_message: str


class SwellSightError(Exception):
    """Base exception class for SwellSight system errors."""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.PROCESSING,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 component: str = "unknown",
                 operation: str = "unknown",
                 recovery_suggestions: Optional[List[str]] = None,
                 user_message: Optional[str] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.component = component
        self.operation = operation
        self.recovery_suggestions = recovery_suggestions or []
        self.user_message = user_message or message
        self.timestamp = time.time()


class InputValidationError(SwellSightError):
    """Error for invalid input data or parameters."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.INPUT_VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class ModelLoadingError(SwellSightError):
    """Error for model loading and initialization failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ProcessingError(SwellSightError):
    """Error for processing and computation failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class MemoryError(SwellSightError):
    """Error for memory-related failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class HardwareError(SwellSightError):
    """Error for hardware-related failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.HARDWARE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ConfigurationError(SwellSightError):
    """Error for configuration and setup failures."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            import random
            jitter_factor = 0.75 + (random.random() * 0.5)
            delay *= jitter_factor
        
        return delay


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.fallback_models: Dict[str, Any] = {}
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self):
        """Setup default recovery strategies for different error categories."""
        self.recovery_strategies = {
            ErrorCategory.MEMORY: [
                self._clear_gpu_cache,
                self._reduce_batch_size,
                self._fallback_to_cpu
            ],
            ErrorCategory.MODEL_LOADING: [
                self._retry_model_loading,
                self._use_fallback_model,
                self._download_model_again
            ],
            ErrorCategory.HARDWARE: [
                self._fallback_to_cpu,
                self._reduce_precision,
                self._use_alternative_backend
            ],
            ErrorCategory.NETWORK: [
                self._retry_with_backoff,
                self._use_cached_model,
                self._use_offline_mode
            ],
            ErrorCategory.FILE_IO: [
                self._retry_file_operation,
                self._check_permissions,
                self._use_alternative_path
            ]
        }
    
    def handle_error(self, 
                    error: Exception,
                    component: str,
                    operation: str,
                    context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle an error with appropriate recovery strategies."""
        
        # Create error context
        error_context = self._create_error_context(
            error, component, operation, context
        )
        
        # Log error
        self._log_error(error_context)
        
        # Store in history
        self.error_history.append(error_context)
        
        # Attempt recovery if strategies available
        if error_context.category in self.recovery_strategies:
            self._attempt_recovery(error_context, context)
        
        return error_context
    
    def _create_error_context(self,
                            error: Exception,
                            component: str,
                            operation: str,
                            context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Create comprehensive error context."""
        
        # Determine error category and severity
        category, severity = self._classify_error(error)
        
        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(
            error, category, context
        )
        
        # Create user-friendly message
        user_message = self._create_user_message(error, category, operation)
        
        return ErrorContext(
            error_type=type(error),
            error_message=str(error),
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            timestamp=time.time(),
            traceback_str=traceback.format_exc(),
            recovery_suggestions=recovery_suggestions,
            user_message=user_message
        )
    
    def _classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity."""
        
        if isinstance(error, SwellSightError):
            return error.category, error.severity
        
        # Classify standard exceptions
        if isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.INPUT_VALIDATION, ErrorSeverity.LOW
        elif isinstance(error, FileNotFoundError):
            return ErrorCategory.FILE_IO, ErrorSeverity.MEDIUM
        elif isinstance(error, PermissionError):
            return ErrorCategory.FILE_IO, ErrorSeverity.HIGH
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        elif isinstance(error, torch.cuda.OutOfMemoryError):
            return ErrorCategory.MEMORY, ErrorSeverity.HIGH
        elif isinstance(error, RuntimeError):
            if "CUDA" in str(error):
                return ErrorCategory.HARDWARE, ErrorSeverity.HIGH
            else:
                return ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM
        else:
            return ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM
    
    def _generate_recovery_suggestions(self,
                                     error: Exception,
                                     category: ErrorCategory,
                                     context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate specific recovery suggestions based on error type."""
        
        suggestions = []
        
        if category == ErrorCategory.INPUT_VALIDATION:
            suggestions.extend([
                "Check input data format and dimensions",
                "Verify image resolution is between 480p and 4K",
                "Ensure image contains ocean content",
                "Validate file format (JPEG, PNG, WebP supported)"
            ])
        
        elif category == ErrorCategory.MEMORY:
            suggestions.extend([
                "Reduce batch size or image resolution",
                "Clear GPU cache and restart processing",
                "Switch to CPU processing if GPU memory insufficient",
                "Close other GPU-intensive applications"
            ])
        
        elif category == ErrorCategory.MODEL_LOADING:
            suggestions.extend([
                "Check internet connection for model download",
                "Verify model files are not corrupted",
                "Clear model cache and re-download",
                "Check available disk space"
            ])
        
        elif category == ErrorCategory.HARDWARE:
            suggestions.extend([
                "Update GPU drivers",
                "Check CUDA installation and compatibility",
                "Switch to CPU processing as fallback",
                "Restart the application"
            ])
        
        elif category == ErrorCategory.FILE_IO:
            suggestions.extend([
                "Check file permissions and access rights",
                "Verify file path exists and is accessible",
                "Ensure sufficient disk space",
                "Check file is not locked by another process"
            ])
        
        elif category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check internet connection",
                "Retry the operation after a short delay",
                "Use cached models if available",
                "Check firewall and proxy settings"
            ])
        
        return suggestions
    
    def _create_user_message(self,
                           error: Exception,
                           category: ErrorCategory,
                           operation: str) -> str:
        """Create user-friendly error message."""
        
        base_messages = {
            ErrorCategory.INPUT_VALIDATION: f"Invalid input for {operation}",
            ErrorCategory.MEMORY: f"Insufficient memory for {operation}",
            ErrorCategory.MODEL_LOADING: f"Failed to load model for {operation}",
            ErrorCategory.HARDWARE: f"Hardware issue during {operation}",
            ErrorCategory.FILE_IO: f"File access error during {operation}",
            ErrorCategory.NETWORK: f"Network error during {operation}",
            ErrorCategory.PROCESSING: f"Processing error during {operation}"
        }
        
        base_message = base_messages.get(category, f"Error during {operation}")
        
        if isinstance(error, SwellSightError) and error.user_message:
            return error.user_message
        
        return f"{base_message}: {str(error)}"
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity."""
        
        log_message = (
            f"[{error_context.component}] {error_context.operation} failed: "
            f"{error_context.error_message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
            logger.critical(f"Traceback: {error_context.traceback_str}")
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
            logger.debug(f"Traceback: {error_context.traceback_str}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log recovery suggestions
        if error_context.recovery_suggestions:
            logger.info("Recovery suggestions:")
            for suggestion in error_context.recovery_suggestions:
                logger.info(f"  - {suggestion}")
    
    def _attempt_recovery(self,
                         error_context: ErrorContext,
                         context: Optional[Dict[str, Any]] = None):
        """Attempt recovery using available strategies."""
        
        strategies = self.recovery_strategies.get(error_context.category, [])
        
        for strategy in strategies:
            try:
                logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                result = strategy(error_context, context)
                if result:
                    logger.info(f"Recovery successful with {strategy.__name__}")
                    break
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
    
    # Recovery strategy implementations
    def _clear_gpu_cache(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Clear GPU cache to free memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
                return True
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
        return False
    
    def _reduce_batch_size(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Reduce batch size in context if available."""
        if context and 'batch_size' in context:
            old_batch_size = context['batch_size']
            context['batch_size'] = max(1, old_batch_size // 2)
            logger.info(f"Reduced batch size from {old_batch_size} to {context['batch_size']}")
            return True
        return False
    
    def _fallback_to_cpu(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Switch processing to CPU."""
        if context and 'device' in context:
            context['device'] = torch.device('cpu')
            logger.info("Switched to CPU processing")
            return True
        return False
    
    def _retry_model_loading(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Retry model loading with exponential backoff."""
        # This would be implemented by the calling code using retry_with_backoff
        return False
    
    def _use_fallback_model(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Use a fallback model if available."""
        if context and 'model_name' in context:
            model_name = context['model_name']
            if model_name in self.fallback_models:
                context['model'] = self.fallback_models[model_name]
                logger.info(f"Using fallback model for {model_name}")
                return True
        return False
    
    def _download_model_again(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Clear model cache and re-download."""
        # This would clear Hugging Face cache and force re-download
        return False
    
    def _reduce_precision(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Reduce model precision to save memory."""
        if context and 'precision' in context:
            context['precision'] = 'fp16' if context['precision'] == 'fp32' else 'int8'
            logger.info(f"Reduced precision to {context['precision']}")
            return True
        return False
    
    def _use_alternative_backend(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Switch to alternative backend."""
        return False
    
    def _retry_with_backoff(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Retry operation with exponential backoff."""
        # This would be handled by the retry_with_backoff decorator
        return False
    
    def _use_cached_model(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Use cached model if available."""
        return False
    
    def _use_offline_mode(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Switch to offline mode."""
        if context:
            context['offline_mode'] = True
            logger.info("Switched to offline mode")
            return True
        return False
    
    def _retry_file_operation(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Retry file operation."""
        return False
    
    def _check_permissions(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Check and fix file permissions."""
        return False
    
    def _use_alternative_path(self, error_context: ErrorContext, context: Optional[Dict] = None) -> bool:
        """Use alternative file path."""
        return False
    
    def register_fallback_model(self, model_name: str, model: Any):
        """Register a fallback model for error recovery."""
        self.fallback_models[model_name] = model
        logger.info(f"Registered fallback model for {model_name}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_history:
            return {"total_errors": 0, "recent_errors": []}
        
        recent_errors = self.error_history[-10:]  # Last 10 errors
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "most_recent": {
                "timestamp": recent_errors[-1].timestamp,
                "component": recent_errors[-1].component,
                "operation": recent_errors[-1].operation,
                "category": recent_errors[-1].category.value,
                "severity": recent_errors[-1].severity.value
            }
        }


# Global error handler instance
error_handler = ErrorHandler()


def retry_with_backoff(retry_config: Optional[RetryConfig] = None,
                      exceptions: tuple = (Exception,),
                      component: str = "unknown",
                      operation: str = "unknown"):
    """Decorator for retry logic with exponential backoff."""
    
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_attempts:
                        # Final attempt failed, handle error
                        error_handler.handle_error(e, component, operation)
                        raise
                    
                    # Calculate delay and wait
                    delay = retry_config.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt}/{retry_config.max_attempts} failed for "
                        f"{operation}: {e}. Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def handle_graceful_degradation(fallback_func: Optional[Callable] = None,
                               component: str = "unknown",
                               operation: str = "unknown"):
    """Decorator for graceful degradation on errors."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle error
                error_context = error_handler.handle_error(e, component, operation)
                
                # Try fallback if available
                if fallback_func:
                    logger.info(f"Using fallback for {operation}")
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                
                # Re-raise if no fallback or fallback failed
                raise
        
        return wrapper
    return decorator


def safe_execute(func: Callable,
                *args,
                component: str = "unknown",
                operation: str = "unknown",
                default_return: Any = None,
                **kwargs) -> Any:
    """Safely execute a function with error handling."""
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.handle_error(e, component, operation)
        logger.warning(f"Safe execution failed for {operation}, returning default")
        return default_return