#!/usr/bin/env python3
"""
Startup script for SwellSight Wave Analysis API.

Handles initialization, health checks, and graceful startup.
"""

import argparse
import logging
import os
import sys
import time
import uvicorn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swellsight.api.server import create_app
from swellsight.api.deployment import create_model_server
from swellsight.utils.logging import setup_logging

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="SwellSight API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for development")
    parser.add_argument("--preload", action="store_true",
                       help="Preload models before starting server")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger("swellsight.startup")
    
    logger.info("Starting SwellSight Wave Analysis API")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Workers: {args.workers}, Log Level: {args.log_level}")
    
    # Preload models if requested
    if args.preload:
        logger.info("Preloading models...")
        try:
            model_server = create_model_server(args.config)
            health = model_server.get_health_status()
            logger.info(f"Model preload status: {health.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"Model preload failed: {e}")
            return 1
    
    # Create FastAPI app
    app = create_app()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30
    )
    
    # Start server
    server = uvicorn.Server(config)
    
    try:
        logger.info("Starting uvicorn server...")
        server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    logger.info("SwellSight API server stopped")
    return 0

if __name__ == "__main__":
    sys.exit(main())