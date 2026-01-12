"""
REST API and serving components for the SwellSight Wave Analysis System.

This module provides FastAPI server, endpoints, and request/response schemas
for production deployment and inference serving.
"""

from .server import create_app
from .endpoints import router
from .schemas import WaveAnalysisRequest, WaveAnalysisResponse

__all__ = [
    "create_app",
    "router",
    "WaveAnalysisRequest",
    "WaveAnalysisResponse"
]