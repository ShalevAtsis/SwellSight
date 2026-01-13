"""
FastAPI server for SwellSight wave analysis API.

Provides REST API endpoints for wave analysis inference and system monitoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import time
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from .endpoints import router
from .deployment import ModelServer
from ..core.pipeline import WaveAnalysisPipeline
from ..utils.logging import setup_logging

# Global model server instance
model_server = None
server_start_time = time.time()
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger = logging.getLogger("swellsight.api")
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with graceful startup and shutdown."""
    global model_server, server_start_time
    
    # Startup
    logger = logging.getLogger("swellsight.api")
    logger.info("Starting SwellSight API server...")
    
    try:
        # Initialize model server
        logger.info("Initializing model server...")
        model_server = ModelServer()
        success = model_server.initialize_models()
        
        if not success:
            logger.warning("Model server initialized with warnings")
        else:
            logger.info("Model server initialized successfully")
        
        # Store model server in app state
        app.state.model_server = model_server
        app.state.pipeline = model_server.pipeline
        app.state.start_time = server_start_time
        
        logger.info("SwellSight API server startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize model server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SwellSight API server...")
    
    try:
        # Cleanup model server resources
        if model_server:
            logger.info("Cleaning up model server resources...")
            model_server.cleanup()
            
        logger.info("SwellSight API server shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    # Clear global state
    model_server = None

def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Create FastAPI app
    app = FastAPI(
        title="SwellSight Wave Analysis API",
        description="AI-powered wave analysis system for surfers",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "service": "SwellSight Wave Analysis API",
            "version": "0.1.0",
            "timestamp": time.time()
        }
    
    # Detailed health check endpoint
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with component status."""
        try:
            # Use model server health status if available
            if hasattr(app.state, 'model_server') and app.state.model_server:
                return app.state.model_server.get_health_status()
            else:
                return {
                    "status": "degraded",
                    "service": "SwellSight Wave Analysis API",
                    "version": "0.1.0",
                    "timestamp": time.time(),
                    "uptime_seconds": time.time() - server_start_time,
                    "error": "Model server not available"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "service": "SwellSight Wave Analysis API",
                "version": "0.1.0",
                "timestamp": time.time(),
                "error": str(e)
            }
    
    # Readiness probe endpoint
    @app.get("/ready")
    async def readiness_check():
        """Readiness probe for container orchestration."""
        try:
            if shutdown_requested:
                raise HTTPException(status_code=503, detail="Shutdown in progress")
            
            # Check if pipeline is ready
            if hasattr(app.state, 'pipeline') and app.state.pipeline:
                status = app.state.pipeline.get_pipeline_status()
                if status.get("components_initialized", False):
                    return {"status": "ready", "timestamp": time.time()}
                else:
                    raise HTTPException(status_code=503, detail="Pipeline not ready")
            else:
                raise HTTPException(status_code=503, detail="Pipeline not available")
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")
    
    # Liveness probe endpoint
    @app.get("/live")
    async def liveness_check():
        """Liveness probe for container orchestration."""
        if shutdown_requested:
            raise HTTPException(status_code=503, detail="Shutdown in progress")
        return {"status": "alive", "timestamp": time.time()}
    
    return app

# Create app instance
app = create_app()