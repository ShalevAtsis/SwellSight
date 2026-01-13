"""
Request and response schemas for SwellSight API.

Defines Pydantic models for API request/response validation and documentation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class WaveAnalysisRequest(BaseModel):
    """Request schema for wave analysis."""
    
    image_url: Optional[str] = Field(None, description="URL of beach cam image to analyze")
    processing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional processing parameters"
    )

class ConfidenceScores(BaseModel):
    """Confidence scores for all predictions."""
    
    height: float = Field(..., ge=0.0, le=1.0, description="Wave height confidence")
    direction: float = Field(..., ge=0.0, le=1.0, description="Wave direction confidence")
    breaking: float = Field(..., ge=0.0, le=1.0, description="Breaking type confidence")
    overall: float = Field(..., ge=0.0, le=1.0, description="Overall prediction confidence")

class WaveAnalysisResponse(BaseModel):
    """Response schema for wave analysis results."""
    
    wave_height_meters: float = Field(..., description="Wave height in meters")
    wave_height_feet: float = Field(..., description="Wave height in feet")
    wave_direction: str = Field(..., description="Wave direction (LEFT/RIGHT/STRAIGHT)")
    breaking_type: str = Field(..., description="Breaking type (SPILLING/PLUNGING/SURGING)")
    confidence_scores: ConfidenceScores = Field(..., description="Confidence scores for all predictions")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    extreme_conditions: bool = Field(..., description="Whether extreme conditions were detected")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")

class BatchAnalysisRequest(BaseModel):
    """Request schema for batch wave analysis."""
    
    image_urls: List[str] = Field(..., description="List of beach cam image URLs")
    processing_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional processing parameters"
    )

class BatchAnalysisResponse(BaseModel):
    """Response schema for batch wave analysis results."""
    
    results: List[WaveAnalysisResponse] = Field(..., description="Analysis results for each image")
    total_images: int = Field(..., description="Total number of images processed")
    successful_analyses: int = Field(..., description="Number of successful analyses")
    failed_analyses: int = Field(..., description="Number of failed analyses")
    total_processing_time_seconds: float = Field(..., description="Total processing time")

class SystemStatusResponse(BaseModel):
    """Response schema for system status."""
    
    status: str = Field(..., description="System status (healthy/degraded/error)")
    components_initialized: bool = Field(..., description="Whether all components are initialized")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: str = Field(..., description="Current memory usage")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Status timestamp")

class ModelInfo(BaseModel):
    """Model information schema."""
    
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (depth/synthetic/analysis)")
    version: str = Field(..., description="Model version")
    loaded: bool = Field(..., description="Whether model is loaded")
    memory_usage_mb: float = Field(..., description="Model memory usage in MB")
    last_updated: datetime = Field(..., description="Last update timestamp")

class ModelsInfoResponse(BaseModel):
    """Response schema for models information."""
    
    models: List[ModelInfo] = Field(..., description="List of model information")
    total_memory_usage_mb: float = Field(..., description="Total memory usage of all models")

class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")