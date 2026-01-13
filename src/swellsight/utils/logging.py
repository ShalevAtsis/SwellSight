"""
Logging setup and configuration for SwellSight system.

Provides structured logging with appropriate levels and formatting
for all system components, including performance metrics collection
and system health monitoring.
"""

import logging
import logging.config
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys
import json
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
import psutil
import torch

@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring."""
    timestamp: float
    component: str
    operation: str
    duration_ms: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float] = None
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    error_occurred: bool = False
    error_message: Optional[str] = None

@dataclass
class SystemHealthMetrics:
    """System health monitoring metrics."""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    disk_usage_percent: float = 0.0
    active_processes: int = 0

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        if hasattr(record, 'memory_usage_mb'):
            log_entry['memory_usage_mb'] = record.memory_usage_mb
        if hasattr(record, 'error_context'):
            log_entry['error_context'] = record.error_context
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

class PerformanceLogger:
    """Logger for performance metrics collection and reporting."""
    
    def __init__(self, logger_name: str = "swellsight.performance"):
        self.logger = logging.getLogger(logger_name)
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        self._lock = threading.Lock()
    
    def log_performance(self, 
                       component: str,
                       operation: str,
                       duration_ms: float,
                       memory_usage_mb: Optional[float] = None,
                       gpu_memory_mb: Optional[float] = None,
                       error_occurred: bool = False,
                       error_message: Optional[str] = None):
        """Log performance metrics for an operation."""
        
        # Get system metrics if not provided
        if memory_usage_mb is None:
            memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        cpu_usage = psutil.cpu_percent()
        gpu_usage = 0.0
        
        # Get GPU metrics if available
        if torch.cuda.is_available():
            try:
                if gpu_memory_mb is None:
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                
                # Try to get GPU utilization
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = float(util.gpu)
                except:
                    gpu_usage = 0.0
            except:
                gpu_memory_mb = None
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_mb=gpu_memory_mb,
            cpu_usage_percent=cpu_usage,
            gpu_usage_percent=gpu_usage,
            error_occurred=error_occurred,
            error_message=error_message
        )
        
        # Store in history
        with self._lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
        
        # Log the metrics
        extra_fields = {
            'component': component,
            'operation': operation,
            'duration_ms': duration_ms,
            'memory_usage_mb': memory_usage_mb,
            'cpu_usage_percent': cpu_usage,
            'gpu_usage_percent': gpu_usage
        }
        
        if gpu_memory_mb is not None:
            extra_fields['gpu_memory_mb'] = gpu_memory_mb
        
        if error_occurred:
            self.logger.error(
                f"Performance: {component}.{operation} completed with error in {duration_ms:.2f}ms",
                extra=extra_fields
            )
        elif duration_ms > 5000:  # Log slow operations as warnings
            self.logger.warning(
                f"Performance: {component}.{operation} slow execution in {duration_ms:.2f}ms",
                extra=extra_fields
            )
        else:
            self.logger.info(
                f"Performance: {component}.{operation} completed in {duration_ms:.2f}ms",
                extra=extra_fields
            )
    
    def get_performance_summary(self, 
                               component: Optional[str] = None,
                               operation: Optional[str] = None,
                               last_n_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for recent operations."""
        
        cutoff_time = time.time() - (last_n_minutes * 60)
        
        with self._lock:
            # Filter metrics
            filtered_metrics = [
                m for m in self.metrics_history
                if m.timestamp >= cutoff_time
                and (component is None or m.component == component)
                and (operation is None or m.operation == operation)
            ]
        
        if not filtered_metrics:
            return {"message": "No metrics found for the specified criteria"}
        
        # Calculate summary statistics
        durations = [m.duration_ms for m in filtered_metrics]
        memory_usage = [m.memory_usage_mb for m in filtered_metrics]
        error_count = sum(1 for m in filtered_metrics if m.error_occurred)
        
        summary = {
            "total_operations": len(filtered_metrics),
            "error_count": error_count,
            "error_rate": error_count / len(filtered_metrics),
            "duration_stats": {
                "min_ms": min(durations),
                "max_ms": max(durations),
                "avg_ms": sum(durations) / len(durations),
                "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 20 else max(durations)
            },
            "memory_stats": {
                "min_mb": min(memory_usage),
                "max_mb": max(memory_usage),
                "avg_mb": sum(memory_usage) / len(memory_usage)
            },
            "time_range": {
                "start": min(m.timestamp for m in filtered_metrics),
                "end": max(m.timestamp for m in filtered_metrics)
            }
        }
        
        return summary

class SystemHealthMonitor:
    """Monitor system health and resource usage."""
    
    def __init__(self, logger_name: str = "swellsight.health"):
        self.logger = logging.getLogger(logger_name)
        self.health_history: List[SystemHealthMetrics] = []
        self.max_history_size = 1000
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = 30  # seconds
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous system health monitoring."""
        if self._monitoring:
            return
        
        self._monitor_interval = interval_seconds
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info(f"Started system health monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous system health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Stopped system health monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self.collect_health_metrics()
                time.sleep(self._monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self._monitor_interval)
    
    def collect_health_metrics(self) -> SystemHealthMetrics:
        """Collect current system health metrics."""
        
        # CPU and memory metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / 1024 / 1024
        memory_usage_percent = memory.percent
        
        # Disk usage
        disk_usage = psutil.disk_usage('/').percent
        
        # Process count
        active_processes = len(psutil.pids())
        
        # GPU metrics
        gpu_available = torch.cuda.is_available()
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        gpu_utilization_percent = None
        
        if gpu_available:
            try:
                gpu_memory_used_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                
                # Try to get GPU utilization
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization_percent = float(util.gpu)
                except:
                    pass
            except:
                pass
        
        # Create health metrics
        health_metrics = SystemHealthMetrics(
            timestamp=time.time(),
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            memory_usage_percent=memory_usage_percent,
            gpu_available=gpu_available,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization_percent=gpu_utilization_percent,
            disk_usage_percent=disk_usage,
            active_processes=active_processes
        )
        
        # Store in history
        with self._lock:
            self.health_history.append(health_metrics)
            if len(self.health_history) > self.max_history_size:
                self.health_history.pop(0)
        
        # Log health metrics
        extra_fields = asdict(health_metrics)
        
        # Determine log level based on resource usage
        if (cpu_usage > 90 or memory_usage_percent > 90 or 
            (gpu_utilization_percent and gpu_utilization_percent > 90)):
            log_level = logging.WARNING
            message = f"High resource usage detected - CPU: {cpu_usage:.1f}%, Memory: {memory_usage_percent:.1f}%"
        elif (cpu_usage > 70 or memory_usage_percent > 70):
            log_level = logging.INFO
            message = f"Moderate resource usage - CPU: {cpu_usage:.1f}%, Memory: {memory_usage_percent:.1f}%"
        else:
            log_level = logging.DEBUG
            message = f"System health normal - CPU: {cpu_usage:.1f}%, Memory: {memory_usage_percent:.1f}%"
        
        self.logger.log(log_level, message, extra=extra_fields)
        
        return health_metrics
    
    def get_health_summary(self, last_n_minutes: int = 60) -> Dict[str, Any]:
        """Get system health summary for recent period."""
        
        cutoff_time = time.time() - (last_n_minutes * 60)
        
        with self._lock:
            filtered_metrics = [
                m for m in self.health_history
                if m.timestamp >= cutoff_time
            ]
        
        if not filtered_metrics:
            return {"message": "No health metrics found for the specified period"}
        
        # Calculate summary statistics
        cpu_usage = [m.cpu_usage_percent for m in filtered_metrics]
        memory_usage = [m.memory_usage_percent for m in filtered_metrics]
        
        summary = {
            "measurement_count": len(filtered_metrics),
            "cpu_stats": {
                "min_percent": min(cpu_usage),
                "max_percent": max(cpu_usage),
                "avg_percent": sum(cpu_usage) / len(cpu_usage)
            },
            "memory_stats": {
                "min_percent": min(memory_usage),
                "max_percent": max(memory_usage),
                "avg_percent": sum(memory_usage) / len(memory_usage)
            },
            "gpu_available": filtered_metrics[-1].gpu_available if filtered_metrics else False,
            "time_range": {
                "start": min(m.timestamp for m in filtered_metrics),
                "end": max(m.timestamp for m in filtered_metrics)
            }
        }
        
        # Add GPU stats if available
        gpu_metrics = [m for m in filtered_metrics if m.gpu_utilization_percent is not None]
        if gpu_metrics:
            gpu_usage = [m.gpu_utilization_percent for m in gpu_metrics]
            summary["gpu_stats"] = {
                "min_percent": min(gpu_usage),
                "max_percent": max(gpu_usage),
                "avg_percent": sum(gpu_usage) / len(gpu_usage)
            }
        
        return summary

# Global instances
performance_logger = PerformanceLogger()
health_monitor = SystemHealthMonitor()

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None,
                 enable_structured_logging: bool = False,
                 enable_performance_logging: bool = True,
                 enable_health_monitoring: bool = True,
                 health_monitor_interval: int = 30) -> None:
    """Setup comprehensive logging configuration for SwellSight system.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional log file path
        log_format: Optional custom log format (ignored if structured logging enabled)
        enable_structured_logging: Enable JSON structured logging
        enable_performance_logging: Enable performance metrics logging
        enable_health_monitoring: Enable system health monitoring
        health_monitor_interval: Health monitoring interval in seconds
    """
    if enable_structured_logging:
        formatter_class = StructuredFormatter
        log_format = None  # StructuredFormatter handles its own format
    else:
        formatter_class = logging.Formatter
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "class": f"{formatter_class.__module__}.{formatter_class.__name__}",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": sys.stdout
            }
        },
        "loggers": {
            "swellsight": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False
            },
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console"]
            }
        }
    }
    
    # Add format to standard formatter if not using structured logging
    if not enable_structured_logging:
        config["formatters"]["standard"]["format"] = log_format
        config["formatters"]["standard"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    
    # Add file handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": log_level,
            "formatter": "standard",
            "filename": str(log_path),
            "mode": "a"
        }
        
        # Add file handler to loggers
        config["loggers"]["swellsight"]["handlers"].append("file")
        config["loggers"][""]["handlers"].append("file")
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Log setup completion
    logger = logging.getLogger("swellsight.logging")
    logger.info(f"Logging setup complete - Level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    if enable_structured_logging:
        logger.info("Structured JSON logging enabled")
    
    # Start performance logging
    if enable_performance_logging:
        logger.info("Performance metrics logging enabled")
    
    # Start health monitoring
    if enable_health_monitoring:
        health_monitor.start_monitoring(health_monitor_interval)
        logger.info(f"System health monitoring enabled with {health_monitor_interval}s interval")

def get_logger(name: str) -> logging.Logger:
    """Get logger instance for specified module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"swellsight.{name}")

class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__)

class PerformanceLoggerMixin:
    """Mixin class to add performance logging capability to any class."""
    
    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics for this component."""
        component = self.__class__.__name__
        performance_logger.log_performance(component, operation, duration_ms, **kwargs)