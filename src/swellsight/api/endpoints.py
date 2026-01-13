"""
API endpoints for wave analysis operations.

Provides REST endpoints for image analysis, batch processing,
and system status monitoring.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import logging
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import lru_cache

from .schemas import (
    WaveAnalysisRequest, WaveAnalysisResponse, SystemStatusResponse,
    BatchAnalysisResponse, ModelsInfoResponse, ModelInfo
)
from ..core.pipeline import WaveAnalysisPipeline

router = APIRouter()
logger = logging.getLogger("swellsight.api.endpoints")

# Simple in-memory cache for results (in production, use Redis or similar)
_result_cache: Dict[str, tuple] = {}
_cache_ttl_seconds = 300  # 5 minutes

def _get_image_hash(image_array: np.ndarray) -> str:
    """Generate hash for image array for caching."""
    return hashlib.md5(image_array.tobytes()).hexdigest()

def _get_cached_result(image_hash: str) -> Optional[WaveAnalysisResponse]:
    """Get cached result if available and not expired."""
    if image_hash in _result_cache:
        result, timestamp = _result_cache[image_hash]
        if time.time() - timestamp < _cache_ttl_seconds:
            logger.info(f"Cache hit for image hash: {image_hash}")
            return result
        else:
            # Remove expired entry
            del _result_cache[image_hash]
    return None

def _cache_result(image_hash: str, result: WaveAnalysisResponse):
    """Cache analysis result."""
    _result_cache[image_hash] = (result, time.time())
    logger.debug(f"Cached result for image hash: {image_hash}")

@lru_cache(maxsize=1)
def _get_pipeline_singleton() -> WaveAnalysisPipeline:
    """Get or create singleton pipeline instance."""
    return WaveAnalysisPipeline()

def get_pipeline() -> WaveAnalysisPipeline:
    """Dependency to get pipeline instance."""
    from fastapi import Request
    from starlette.requests import Request as StarletteRequest
    import inspect
    
    # Get the current request context to access app state
    frame = inspect.currentframe()
    try:
        while frame:
            if 'request' in frame.f_locals and hasattr(frame.f_locals['request'], 'app'):
                request = frame.f_locals['request']
                if hasattr(request.app.state, 'pipeline'):
                    return request.app.state.pipeline
            frame = frame.f_back
    finally:
        del frame
    
    # Fallback: create new pipeline instance if not found in app state
    logger.warning("Pipeline not found in app state, creating new instance")
    return WaveAnalysisPipeline()

@router.post("/analyze", response_model=WaveAnalysisResponse)
async def analyze_wave_image(
    file: UploadFile = File(...),
    pipeline: WaveAnalysisPipeline = Depends(get_pipeline)
):
    """Analyze wave conditions from uploaded beach cam image.
    
    Args:
        file: Uploaded image file
        pipeline: Wave analysis pipeline instance
        
    Returns:
        Wave analysis results with metrics and confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image.convert('RGB'))
        
        # Check cache first
        image_hash = _get_image_hash(image_array)
        cached_result = _get_cached_result(image_hash)
        if cached_result:
            return cached_result
        
        # Process through pipeline
        results = pipeline.process_beach_cam_image(image_array)
        
        # Convert to response format
        response = WaveAnalysisResponse(
            wave_height_meters=results.wave_metrics.height_meters,
            wave_height_feet=results.wave_metrics.height_feet,
            wave_direction=results.wave_metrics.direction,
            breaking_type=results.wave_metrics.breaking_type,
            confidence_scores={
                "height": results.wave_metrics.height_confidence,
                "direction": results.wave_metrics.direction_confidence,
                "breaking": results.wave_metrics.breaking_confidence,
                "overall": results.pipeline_confidence
            },
            processing_time_seconds=results.processing_time,
            extreme_conditions=results.wave_metrics.extreme_conditions,
            warnings=results.warnings
        )
        
        # Cache the result
        _cache_result(image_hash, response)
        
        logger.info(f"Successfully analyzed image: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to analyze image {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch_images(
    files: List[UploadFile] = File(...),
    pipeline: WaveAnalysisPipeline = Depends(get_pipeline)
):
    """Analyze multiple beach cam images in batch.
    
    Args:
        files: List of uploaded image files
        pipeline: Wave analysis pipeline instance
        
    Returns:
        Batch analysis results with individual results and summary
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 50 images")
    
    results = []
    successful_analyses = 0
    failed_analyses = 0
    start_time = time.time()
    
    # Process images in batch
    image_arrays = []
    filenames = []
    
    try:
        # Read all images first
        for file in files:
            if not file.content_type.startswith('image/'):
                logger.warning(f"Skipping non-image file: {file.filename}")
                failed_analyses += 1
                continue
                
            try:
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                image_array = np.array(image.convert('RGB'))
                image_arrays.append(image_array)
                filenames.append(file.filename)
            except Exception as e:
                logger.error(f"Failed to read image {file.filename}: {e}")
                failed_analyses += 1
                continue
        
        # Process batch through pipeline
        if image_arrays:
            batch_results = pipeline.process_batch_images(image_arrays)
            
            for i, (result, filename) in enumerate(zip(batch_results, filenames)):
                if result is not None:
                    response = WaveAnalysisResponse(
                        wave_height_meters=result.wave_metrics.height_meters,
                        wave_height_feet=result.wave_metrics.height_feet,
                        wave_direction=result.wave_metrics.direction,
                        breaking_type=result.wave_metrics.breaking_type,
                        confidence_scores={
                            "height": result.wave_metrics.height_confidence,
                            "direction": result.wave_metrics.direction_confidence,
                            "breaking": result.wave_metrics.breaking_confidence,
                            "overall": result.pipeline_confidence
                        },
                        processing_time_seconds=result.processing_time,
                        extreme_conditions=result.wave_metrics.extreme_conditions,
                        warnings=result.warnings
                    )
                    results.append(response)
                    successful_analyses += 1
                    logger.info(f"Successfully analyzed batch image: {filename}")
                else:
                    logger.error(f"Failed to analyze batch image: {filename}")
                    failed_analyses += 1
        
        total_processing_time = time.time() - start_time
        
        return BatchAnalysisResponse(
            results=results,
            total_images=len(files),
            successful_analyses=successful_analyses,
            failed_analyses=failed_analyses,
            total_processing_time_seconds=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    pipeline: WaveAnalysisPipeline = Depends(get_pipeline)
):
    """Get current system status and health metrics.
    
    Args:
        pipeline: Wave analysis pipeline instance
        
    Returns:
        System status information
    """
    try:
        status = pipeline.get_pipeline_status()
        
        return SystemStatusResponse(
            status="healthy" if status["components_initialized"] else "degraded",
            components_initialized=status["components_initialized"],
            gpu_available=status["gpu_available"],
            memory_usage=status.get("memory_usage", "Unknown"),
            uptime_seconds=0,  # TODO: Track actual uptime
            version="0.1.0"
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@router.get("/models/info", response_model=ModelsInfoResponse)
async def get_model_info(
    pipeline: WaveAnalysisPipeline = Depends(get_pipeline)
):
    """Get information about loaded models and their configurations.
    
    Args:
        pipeline: Wave analysis pipeline instance
        
    Returns:
        Information about all loaded models
    """
    try:
        models = []
        total_memory_usage = 0.0
        
        # Get pipeline status to check component initialization
        status = pipeline.get_pipeline_status()
        
        # Depth extraction model info
        if hasattr(pipeline, 'depth_extractor') and pipeline.depth_extractor is not None:
            depth_memory = getattr(pipeline.depth_extractor, 'memory_usage_mb', 0.0)
            models.append(ModelInfo(
                model_name="Depth Anything V2",
                model_type="depth",
                version="2.0",
                loaded=status.get("depth_extractor_initialized", False),
                memory_usage_mb=depth_memory,
                last_updated=datetime.now()
            ))
            total_memory_usage += depth_memory
        
        # Wave analyzer model info
        if hasattr(pipeline, 'wave_analyzer') and pipeline.wave_analyzer is not None:
            analyzer_memory = getattr(pipeline.wave_analyzer, 'memory_usage_mb', 0.0)
            models.append(ModelInfo(
                model_name="Multi-Task Wave Analyzer",
                model_type="analysis",
                version="1.0",
                loaded=status.get("wave_analyzer_initialized", False),
                memory_usage_mb=analyzer_memory,
                last_updated=datetime.now()
            ))
            total_memory_usage += analyzer_memory
        
        # Synthetic generator model info
        if hasattr(pipeline, 'synthetic_generator') and pipeline.synthetic_generator is not None:
            synthetic_memory = getattr(pipeline.synthetic_generator, 'memory_usage_mb', 0.0)
            models.append(ModelInfo(
                model_name="FLUX ControlNet",
                model_type="synthetic",
                version="1.0",
                loaded=status.get("synthetic_generator_initialized", False),
                memory_usage_mb=synthetic_memory,
                last_updated=datetime.now()
            ))
            total_memory_usage += synthetic_memory
        
        return ModelsInfoResponse(
            models=models,
            total_memory_usage_mb=total_memory_usage
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/models/reload")
async def reload_models(
    pipeline: WaveAnalysisPipeline = Depends(get_pipeline)
):
    """Reload models (for updates without server restart).
    
    Args:
        pipeline: Wave analysis pipeline instance
        
    Returns:
        Status of model reload operation
    """
    try:
        logger.info("Starting model reload operation")
        
        # Get current status before reload
        old_status = pipeline.get_pipeline_status()
        
        # Attempt to reload pipeline components
        reload_success = True
        reload_details = {}
        
        try:
            # Reinitialize the pipeline
            pipeline._initialize_components()
            reload_details["pipeline_reinitialized"] = True
            logger.info("Pipeline components reinitialized successfully")
        except Exception as e:
            reload_success = False
            reload_details["pipeline_error"] = str(e)
            logger.error(f"Failed to reinitialize pipeline: {e}")
        
        # Get new status after reload
        new_status = pipeline.get_pipeline_status()
        
        return {
            "status": "success" if reload_success else "partial_failure",
            "message": "Models reloaded successfully" if reload_success else "Model reload completed with errors",
            "reload_details": reload_details,
            "old_status": old_status,
            "new_status": new_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")