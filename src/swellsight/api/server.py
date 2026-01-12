"""
FastAPI server for SwellSight wave analysis API.

Provides REST API endpoints for wave analysis inference and system monitoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
from contextlib import asynccontextmanager

from .endpoints import router
from ..core.pipeline import WaveAnalysisPipeline
from ..utils.logging import setup_logging

# Global pipeline instance
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global pipeline
    
    # Startup
    logger = logging.getLogger("swellsight.api")
    logger.info("Starting SwellSight API server...")
    
    try:
        # Initialize wave analysis pipeline
        pipeline = WaveAnalysisPipeline()
        logger.info("Wave analysis pipeline initialized")
        
        # Store pipeline in app state
        app.state.pipeline = pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SwellSight API server...")
    # Cleanup resources if needed

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
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "SwellSight Wave Analysis API",
            "version": "0.1.0"
        }
    
    return app

# Create app instance
app = create_app()