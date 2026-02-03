#!/usr/bin/env python3
"""
SwellSight Real-time Monitoring Dashboard

Creates visualizations for monitoring model performance in production.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import json

def create_realtime_monitoring_plots(output_dir: Path):
    """Create real-time monitoring visualizations."""
    
    # Generate sample real-time data
    now = datetime.now()
    times = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    
    # Inference metrics over time
    inference_times = [45 + 10 * np.sin(i/4) + np.random.normal(0, 3) for i in range(24)]
    throughput = [22 + 3 * np.sin(i/3) + np.random.normal(0, 1) for i in range(24)]
    memory_usage = [1024 + 100 * np.sin(i/5) + np.random.normal(0, 20) for i in range(24)]
    
    # Accuracy metrics over time
    height_accuracy = [0.87 + 0.05 * np.sin(i/6) + np.random.normal(0, 0.02) for i in range(24)]
    direction_accuracy = [0.92 + 0.03 * np.sin(i/4) + np.random.normal(0, 0.015) for i in range(24)]
    
    # Error rates
    error_rates = [0.02 + 0.01 * np.sin(i/8) + np.random.normal(0, 0.005) for i in range(24)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('SwellSight Real-time Monitoring Dashboard', fontsize=16, fontweight='bold')
    
    # Performance metrics over time
    ax1.plot(times, inference_times, 'b-', linewidth=2, label='Inference Time (ms)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(times, throughput, 'r-', linewidth=2, label='Throughput (FPS)')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Inference Time (ms)', color='b')
    ax1_twin.set_ylabel('Throughput (FPS)', color='r')
    ax1.set_title('Performance Metrics Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Memory usage
    ax2.plot(times, memory_usage, 'g-', linewidth=2)
    ax2.fill_between(times, memory_usage, alpha=0.3, color='green')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.tick_params(axis='x', rotation=45)
    
    # Accuracy metrics
    ax3.plot(times, height_accuracy, 'purple', linewidth=2, label='Height Accuracy')
    ax3.plot(times, direction_accuracy, 'orange', linewidth=2, label='Direction Accuracy')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Model Accuracy Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.8, 1.0)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.tick_params(axis='x', rotation=45)
    
    # Error rates and alerts
    ax4.plot(times, error_rates, 'red', linewidth=2)
    ax4.fill_between(times, error_rates, alpha=0.3, color='red')
    ax4.axhline(y=0.05, color='orange', linestyle='--', label='Warning Threshold')
    ax4.axhline(y=0.1, color='red', linestyle='--', label='Critical Threshold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Error Rate')
    ax4.set_title('Error Rate Monitoring')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'realtime_monitoring.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_system_health_dashboard(output_dir: Path):
    """Create system health monitoring dashboard."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SwellSight System Health Dashboard', fontsize=16, fontweight='bold')
    
    # CPU and GPU utilization
    time_points = list(range(0, 60, 5))  # Last 60 minutes, every 5 minutes
    cpu_usage = [40 + 20 * np.sin(i/10) + np.random.normal(0, 5) for i in time_points]
    gpu_usage = [60 + 25 * np.sin(i/8) + np.random.normal(0, 8) for i in time_points]
    
    ax1.plot(time_points, cpu_usage, 'b-', linewidth=2, label='CPU Usage (%)')
    ax1.plot(time_points, gpu_usage, 'r-', linewidth=2, label='GPU Usage (%)')
    ax1.fill_between(time_points, cpu_usage, alpha=0.3, color='blue')
    ax1.fill_between(time_points, gpu_usage, alpha=0.3, color='red')
    ax1.set_xlabel('Time (minutes ago)')
    ax1.set_ylabel('Usage (%)')
    ax1.set_title('CPU & GPU Utilization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 100)
    
    # Request volume and response times
    request_volume = [50 + 30 * np.sin(i/15) + np.random.normal(0, 10) for i in time_points]
    response_times = [45 + 15 * np.sin(i/12) + np.random.normal(0, 5) for i in time_points]
    
    ax2_twin = ax2.twinx()
    bars = ax2.bar(time_points, request_volume, width=3, alpha=0.6, color='green', label='Requests/min')
    line = ax2_twin.plot(time_points, response_times, 'orange', linewidth=2, marker='o', label='Response Time (ms)')
    
    ax2.set_xlabel('Time (minutes ago)')
    ax2.set_ylabel('Requests per minute', color='green')
    ax2_twin.set_ylabel('Response Time (ms)', color='orange')
    ax2.set_title('Request Volume & Response Times')
    ax2.grid(True, alpha=0.3)
    
    # Model prediction confidence distribution
    confidence_scores = np.random.beta(3, 1, 1000)  # Simulated confidence scores
    ax3.hist(confidence_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.mean(confidence_scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(confidence_scores):.3f}')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Prediction Confidence Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # System alerts and status
    alert_types = ['Memory\nWarning', 'High\nLatency', 'Low\nAccuracy', 'GPU\nOverheat', 'Disk\nSpace']
    alert_counts = [2, 1, 0, 0, 1]
    colors = ['orange' if count > 0 else 'green' for count in alert_counts]
    
    bars = ax4.bar(alert_types, alert_counts, color=colors, alpha=0.8)
    ax4.set_xlabel('Alert Type')
    ax4.set_ylabel('Active Alerts')
    ax4.set_title('System Alerts Status')
    ax4.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, alert_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'system_health.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_deployment_metrics(output_dir: Path):
    """Create deployment and production metrics visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SwellSight Deployment & Production Metrics', fontsize=16, fontweight='bold')
    
    # Model versions in production
    versions = ['v2.0', 'v2.1', 'v2.2-beta']
    traffic_split = [20, 75, 5]  # Percentage of traffic
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    wedges, texts, autotexts = ax1.pie(traffic_split, labels=versions, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax1.set_title('Traffic Split Across Model Versions')
    
    # Geographic distribution of requests
    regions = ['North\nAmerica', 'Europe', 'Asia\nPacific', 'South\nAmerica', 'Others']
    request_counts = [450, 320, 280, 120, 80]
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    bars = ax2.bar(regions, request_counts, color=colors, alpha=0.8)
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Requests (thousands)')
    ax2.set_title('Geographic Request Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, request_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}k', ha='center', va='bottom', fontweight='bold')
    
    # API endpoint usage
    endpoints = ['/predict', '/batch', '/health', '/metrics', '/status']
    usage_counts = [850, 120, 50, 30, 25]
    
    ax3.barh(endpoints, usage_counts, color='teal', alpha=0.8)
    ax3.set_xlabel('Requests (thousands)')
    ax3.set_ylabel('API Endpoint')
    ax3.set_title('API Endpoint Usage')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, count in enumerate(usage_counts):
        ax3.text(count + 20, i, f'{count}k', va='center', fontweight='bold')
    
    # Cost and efficiency metrics
    metrics = ['Compute\nCost', 'Storage\nCost', 'Bandwidth\nCost', 'Total\nCost']
    costs = [1200, 300, 150, 1650]  # USD per month
    colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD']
    
    bars = ax4.bar(metrics, costs, color=colors, alpha=0.8)
    ax4.set_xlabel('Cost Category')
    ax4.set_ylabel('Cost (USD/month)')
    ax4.set_title('Monthly Operational Costs')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 30,
                f'${cost}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'deployment_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate monitoring visualizations."""
    print("📊 SwellSight Monitoring Dashboard Generator")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "evaluation_reports" / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("⏱️  Creating real-time monitoring dashboard...")
    create_realtime_monitoring_plots(output_dir)
    
    print("🏥 Creating system health dashboard...")
    create_system_health_dashboard(output_dir)
    
    print("🚀 Creating deployment metrics...")
    create_deployment_metrics(output_dir)
    
    print("\n✅ Monitoring dashboard generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print("\nGenerated monitoring files:")
    for file in output_dir.glob("*.png"):
        print(f"  📈 {file.name}")

if __name__ == "__main__":
    main()