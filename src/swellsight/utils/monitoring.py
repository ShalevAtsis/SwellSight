"""
System monitoring and alerting for SwellSight.

Provides comprehensive monitoring of system health, performance metrics,
and automated alerting for critical issues.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    MimeText = None
    MimeMultipart = None

from .logging import performance_logger, health_monitor, SystemHealthMetrics, PerformanceMetrics
from .error_handler import error_handler, ErrorSeverity, ErrorCategory

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ERROR_RATE_HIGH = "error_rate_high"
    SYSTEM_HEALTH = "system_health"
    MODEL_FAILURE = "model_failure"
    MEMORY_LEAK = "memory_leak"

@dataclass
class Alert:
    """System alert with details and context."""
    timestamp: float
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    component: str
    metrics: Dict[str, Any]
    suggested_actions: List[str]
    alert_id: str

@dataclass
class AlertRule:
    """Configuration for alert conditions."""
    alert_type: AlertType
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    title_template: str
    message_template: str
    suggested_actions: List[str]
    cooldown_minutes: int = 15  # Minimum time between same alerts

class SystemMonitor:
    """Comprehensive system monitoring with alerting."""
    
    def __init__(self, 
                 alert_config_file: Optional[str] = None,
                 enable_email_alerts: bool = False,
                 email_config: Optional[Dict[str, str]] = None):
        """Initialize system monitor.
        
        Args:
            alert_config_file: Path to alert configuration file
            enable_email_alerts: Whether to send email alerts
            email_config: Email configuration (smtp_server, port, username, password, recipients)
        """
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_cooldowns: Dict[str, float] = {}
        self.max_history_size = 1000
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = 60  # seconds
        self._lock = threading.Lock()
        
        # Email alerting
        self.enable_email_alerts = enable_email_alerts
        self.email_config = email_config or {}
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Load custom alert rules if provided
        if alert_config_file:
            self._load_alert_config(alert_config_file)
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for common issues."""
        
        # High CPU usage
        self.alert_rules.append(AlertRule(
            alert_type=AlertType.RESOURCE_EXHAUSTION,
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: metrics.get('cpu_usage_percent', 0) > 85,
            title_template="High CPU Usage Detected",
            message_template="CPU usage is {cpu_usage_percent:.1f}%, exceeding 85% threshold",
            suggested_actions=[
                "Check for runaway processes",
                "Consider scaling up compute resources",
                "Review recent code changes for performance issues"
            ]
        ))
        
        # High memory usage
        self.alert_rules.append(AlertRule(
            alert_type=AlertType.RESOURCE_EXHAUSTION,
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: metrics.get('memory_usage_percent', 0) > 85,
            title_template="High Memory Usage Detected",
            message_template="Memory usage is {memory_usage_percent:.1f}%, exceeding 85% threshold",
            suggested_actions=[
                "Check for memory leaks",
                "Clear GPU cache if using CUDA",
                "Restart the application",
                "Consider increasing available memory"
            ]
        ))
        
        # Critical memory usage
        self.alert_rules.append(AlertRule(
            alert_type=AlertType.RESOURCE_EXHAUSTION,
            severity=AlertSeverity.CRITICAL,
            condition=lambda metrics: metrics.get('memory_usage_percent', 0) > 95,
            title_template="Critical Memory Usage",
            message_template="Memory usage is {memory_usage_percent:.1f}%, system may become unstable",
            suggested_actions=[
                "Immediately restart the application",
                "Kill non-essential processes",
                "Check for memory leaks",
                "Scale up memory resources"
            ]
        ))
        
        # High error rate
        self.alert_rules.append(AlertRule(
            alert_type=AlertType.ERROR_RATE_HIGH,
            severity=AlertSeverity.ERROR,
            condition=lambda metrics: metrics.get('error_rate', 0) > 0.1,  # 10% error rate
            title_template="High Error Rate Detected",
            message_template="Error rate is {error_rate:.1%}, exceeding 10% threshold",
            suggested_actions=[
                "Check application logs for error patterns",
                "Verify input data quality",
                "Check model loading and initialization",
                "Review recent deployments"
            ]
        ))
        
        # Slow performance
        self.alert_rules.append(AlertRule(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: metrics.get('avg_duration_ms', 0) > 10000,  # 10 seconds
            title_template="Performance Degradation Detected",
            message_template="Average operation time is {avg_duration_ms:.0f}ms, exceeding 10s threshold",
            suggested_actions=[
                "Check system resource usage",
                "Review recent code changes",
                "Consider optimizing model inference",
                "Check for network latency issues"
            ]
        ))
        
        # GPU memory issues
        self.alert_rules.append(AlertRule(
            alert_type=AlertType.RESOURCE_EXHAUSTION,
            severity=AlertSeverity.ERROR,
            condition=lambda metrics: (
                metrics.get('gpu_available', False) and 
                metrics.get('gpu_memory_used_mb', 0) / max(metrics.get('gpu_memory_total_mb', 1), 1) > 0.9
            ),
            title_template="GPU Memory Exhaustion",
            message_template="GPU memory usage is {gpu_memory_usage_percent:.1f}%, exceeding 90% threshold",
            suggested_actions=[
                "Clear GPU cache with torch.cuda.empty_cache()",
                "Reduce batch size or model size",
                "Switch to CPU processing temporarily",
                "Restart the application"
            ]
        ))
    
    def _load_alert_config(self, config_file: str):
        """Load alert configuration from file."""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Parse custom alert rules
                for rule_config in config.get('alert_rules', []):
                    # This would need more sophisticated parsing for custom conditions
                    logger.info(f"Custom alert rule loaded: {rule_config.get('title', 'Unknown')}")
                
                logger.info(f"Loaded alert configuration from {config_file}")
            else:
                logger.warning(f"Alert config file not found: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load alert configuration: {e}")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous system monitoring."""
        if self._monitoring:
            return
        
        self._monitor_interval = interval_seconds
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started system monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped system monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._check_alerts()
                time.sleep(self._monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self._monitor_interval)
    
    def _check_alerts(self):
        """Check all alert conditions and trigger alerts if needed."""
        current_time = time.time()
        
        # Get current system health
        health_metrics = health_monitor.collect_health_metrics()
        health_dict = asdict(health_metrics)
        
        # Get performance summary
        perf_summary = performance_logger.get_performance_summary(last_n_minutes=10)
        
        # Combine metrics
        combined_metrics = {**health_dict, **perf_summary}
        
        # Add calculated metrics
        if health_metrics.gpu_memory_used_mb and health_metrics.gpu_memory_total_mb:
            combined_metrics['gpu_memory_usage_percent'] = (
                health_metrics.gpu_memory_used_mb / health_metrics.gpu_memory_total_mb * 100
            )
        
        if 'duration_stats' in perf_summary:
            combined_metrics['avg_duration_ms'] = perf_summary['duration_stats'].get('avg_ms', 0)
        
        # Check each alert rule
        for rule in self.alert_rules:
            try:
                if rule.condition(combined_metrics):
                    self._trigger_alert(rule, combined_metrics)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule.alert_type}: {e}")
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert if not in cooldown period."""
        current_time = time.time()
        alert_key = f"{rule.alert_type.value}_{rule.severity.value}"
        
        # Check cooldown
        if alert_key in self.alert_cooldowns:
            if current_time - self.alert_cooldowns[alert_key] < rule.cooldown_minutes * 60:
                return  # Still in cooldown
        
        # Create alert
        alert_id = f"{alert_key}_{int(current_time)}"
        title = rule.title_template.format(**metrics)
        message = rule.message_template.format(**metrics)
        
        alert = Alert(
            timestamp=current_time,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=title,
            message=message,
            component="SystemMonitor",
            metrics=metrics.copy(),
            suggested_actions=rule.suggested_actions.copy(),
            alert_id=alert_id
        )
        
        # Store alert
        with self._lock:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history_size:
                self.alert_history.pop(0)
        
        # Set cooldown
        self.alert_cooldowns[alert_key] = current_time
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(rule.severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT: {title} - {message}")
        
        # Send email alert if enabled
        if self.enable_email_alerts:
            self._send_email_alert(alert)
        
        # Handle critical alerts
        if rule.severity == AlertSeverity.CRITICAL:
            self._handle_critical_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert notification."""
        if not EMAIL_AVAILABLE:
            logger.warning("Email functionality not available, skipping email alert")
            return
            
        try:
            if not self.email_config.get('smtp_server'):
                return
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.email_config.get('username', 'swellsight@system')
            msg['To'] = ', '.join(self.email_config.get('recipients', []))
            msg['Subject'] = f"SwellSight Alert: {alert.title}"
            
            # Email body
            body = f"""
SwellSight System Alert

Alert Type: {alert.alert_type.value}
Severity: {alert.severity.value}
Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}
Component: {alert.component}

Message: {alert.message}

Suggested Actions:
{chr(10).join(f"- {action}" for action in alert.suggested_actions)}

System Metrics:
{json.dumps(alert.metrics, indent=2)}

Alert ID: {alert.alert_id}
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(
                self.email_config['smtp_server'], 
                self.email_config.get('port', 587)
            )
            server.starttls()
            server.login(
                self.email_config['username'], 
                self.email_config['password']
            )
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _handle_critical_alert(self, alert: Alert):
        """Handle critical alerts with immediate actions."""
        logger.critical(f"CRITICAL ALERT: {alert.title}")
        
        # Take immediate action based on alert type
        if alert.alert_type == AlertType.RESOURCE_EXHAUSTION:
            if 'memory' in alert.message.lower():
                # Try to free memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared GPU cache due to critical memory alert")
                except:
                    pass
                
                # Force garbage collection
                import gc
                gc.collect()
                logger.info("Forced garbage collection due to critical memory alert")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of currently active alerts."""
        with self._lock:
            return self.active_alerts.copy()
    
    def get_alert_history(self, last_n_hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period."""
        cutoff_time = time.time() - (last_n_hours * 3600)
        
        with self._lock:
            return [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_time
            ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""
        with self._lock:
            for i, alert in enumerate(self.active_alerts):
                if alert.alert_id == alert_id:
                    self.active_alerts.pop(i)
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        current_time = time.time()
        
        # Get current metrics
        health_metrics = health_monitor.collect_health_metrics()
        perf_summary = performance_logger.get_performance_summary(last_n_minutes=60)
        error_summary = error_handler.get_error_summary()
        
        # Count alerts by severity
        active_alerts = self.get_active_alerts()
        alert_counts = {
            "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "error": len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
            "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            "info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
        }
        
        # Determine overall system health
        if alert_counts["critical"] > 0:
            overall_health = "critical"
        elif alert_counts["error"] > 0:
            overall_health = "degraded"
        elif alert_counts["warning"] > 0:
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        return {
            "timestamp": current_time,
            "overall_health": overall_health,
            "system_metrics": asdict(health_metrics),
            "performance_summary": perf_summary,
            "error_summary": error_summary,
            "active_alerts": {
                "total": len(active_alerts),
                "by_severity": alert_counts
            },
            "monitoring_status": {
                "is_monitoring": self._monitoring,
                "monitor_interval": self._monitor_interval
            }
        }

# Global system monitor instance
system_monitor = SystemMonitor()