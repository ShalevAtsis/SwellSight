#!/usr/bin/env python3
"""
Deployment script for SwellSight Wave Analysis API.

Provides utilities for deploying, managing, and monitoring the API server.
"""

import argparse
import json
import sys
import time
import requests
from pathlib import Path

def check_health(base_url: str, timeout: int = 30) -> bool:
    """Check if the API server is healthy.
    
    Args:
        base_url: Base URL of the API server
        timeout: Timeout in seconds
        
    Returns:
        True if healthy, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/health/detailed", timeout=timeout)
        if response.status_code == 200:
            health_data = response.json()
            print(f"Server status: {health_data.get('status', 'unknown')}")
            print(f"Uptime: {health_data.get('uptime_seconds', 0):.1f} seconds")
            return health_data.get('status') in ['healthy', 'degraded']
        else:
            print(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def wait_for_ready(base_url: str, max_wait: int = 120) -> bool:
    """Wait for the API server to be ready.
    
    Args:
        base_url: Base URL of the API server
        max_wait: Maximum wait time in seconds
        
    Returns:
        True if ready, False if timeout
    """
    print(f"Waiting for server to be ready (max {max_wait}s)...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{base_url}/ready", timeout=5)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except:
            pass
        
        print(".", end="", flush=True)
        time.sleep(2)
    
    print("\nTimeout waiting for server to be ready")
    return False

def reload_models(base_url: str) -> bool:
    """Reload models on the running server.
    
    Args:
        base_url: Base URL of the API server
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print("Reloading models...")
        response = requests.post(f"{base_url}/api/v1/models/reload", timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Reload status: {result.get('status', 'unknown')}")
            print(f"Message: {result.get('message', 'No message')}")
            return result.get('status') == 'success'
        else:
            print(f"Model reload failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Model reload failed: {e}")
        return False

def get_model_info(base_url: str) -> bool:
    """Get information about loaded models.
    
    Args:
        base_url: Base URL of the API server
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/v1/models/info", timeout=30)
        
        if response.status_code == 200:
            info = response.json()
            print("Model Information:")
            print(f"Total memory usage: {info.get('total_memory_usage_mb', 0):.1f} MB")
            print("\nLoaded models:")
            
            for model in info.get('models', []):
                print(f"  - {model.get('model_name', 'Unknown')}")
                print(f"    Type: {model.get('model_type', 'Unknown')}")
                print(f"    Loaded: {model.get('loaded', False)}")
                print(f"    Memory: {model.get('memory_usage_mb', 0):.1f} MB")
                print()
            
            return True
        else:
            print(f"Failed to get model info with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Failed to get model info: {e}")
        return False

def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="SwellSight API Deployment Manager")
    parser.add_argument("--base-url", default="http://localhost:8000", 
                       help="Base URL of the API server")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check server health")
    health_parser.add_argument("--timeout", type=int, default=30, 
                              help="Timeout in seconds")
    
    # Wait for ready command
    ready_parser = subparsers.add_parser("wait-ready", help="Wait for server to be ready")
    ready_parser.add_argument("--max-wait", type=int, default=120, 
                             help="Maximum wait time in seconds")
    
    # Reload models command
    subparsers.add_parser("reload", help="Reload models")
    
    # Model info command
    subparsers.add_parser("models", help="Get model information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    base_url = args.base_url.rstrip('/')
    
    if args.command == "health":
        success = check_health(base_url, args.timeout)
        return 0 if success else 1
    
    elif args.command == "wait-ready":
        success = wait_for_ready(base_url, args.max_wait)
        return 0 if success else 1
    
    elif args.command == "reload":
        success = reload_models(base_url)
        return 0 if success else 1
    
    elif args.command == "models":
        success = get_model_info(base_url)
        return 0 if success else 1
    
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())