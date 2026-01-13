"""
Tests for API deployment and inference interface.

Tests the production inference API and deployment utilities.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io

from src.swellsight.api.server import create_app
from src.swellsight.api.deployment import ModelServer
from src.swellsight.core.pipeline import WaveAnalysisPipeline

class TestProductionInferenceAPI:
    """Test production inference API functionality."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline for testing."""
        pipeline = Mock(spec=WaveAnalysisPipeline)
        
        # Mock pipeline status
        pipeline.get_pipeline_status.return_value = {
            "components_initialized": True,
            "gpu_available": True,
            "memory_usage": "2.5GB"
        }
        
        # Mock analysis result
        mock_result = Mock()
        mock_result.wave_metrics.height_meters = 1.5
        mock_result.wave_metrics.height_feet = 4.9
        mock_result.wave_metrics.direction = "LEFT"
        mock_result.wave_metrics.breaking_type = "SPILLING"
        mock_result.wave_metrics.height_confidence = 0.85
        mock_result.wave_metrics.direction_confidence = 0.90
        mock_result.wave_metrics.breaking_confidence = 0.88
        mock_result.wave_metrics.extreme_conditions = False
        mock_result.pipeline_confidence = 0.87
        mock_result.processing_time = 0.15
        mock_result.warnings = []
        
        pipeline.process_beach_cam_image.return_value = mock_result
        pipeline.process_batch_images.return_value = [mock_result, mock_result]
        
        return pipeline
    
    @pytest.fixture
    def test_client(self, mock_pipeline):
        """Create test client with mocked pipeline."""
        app = create_app()
        
        # Mock the pipeline in app state
        app.state.pipeline = mock_pipeline
        
        return TestClient(app)
    
    def test_health_endpoint(self, test_client):
        """Test basic health endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "SwellSight Wave Analysis API"
        assert "timestamp" in data
    
    def test_detailed_health_endpoint(self, test_client):
        """Test detailed health endpoint."""
        response = test_client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
    
    def test_readiness_endpoint(self, test_client):
        """Test readiness probe endpoint."""
        response = test_client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ready"
    
    def test_liveness_endpoint(self, test_client):
        """Test liveness probe endpoint."""
        response = test_client.get("/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
    
    def test_single_image_analysis(self, test_client):
        """Test single image analysis endpoint."""
        # Create test image
        image = Image.new('RGB', (640, 480), color='blue')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = test_client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "wave_height_meters" in data
        assert "wave_height_feet" in data
        assert "wave_direction" in data
        assert "breaking_type" in data
        assert "confidence_scores" in data
        assert "processing_time_seconds" in data
        
        # Verify confidence scores structure
        confidence = data["confidence_scores"]
        assert "height" in confidence
        assert "direction" in confidence
        assert "breaking" in confidence
        assert "overall" in confidence
    
    def test_batch_image_analysis(self, test_client):
        """Test batch image analysis endpoint."""
        # Create test images
        images = []
        for i in range(2):
            image = Image.new('RGB', (640, 480), color='blue')
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images.append(("file", (f"test{i}.jpg", img_bytes, "image/jpeg")))
        
        response = test_client.post("/api/v1/analyze/batch", files=images)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify batch response structure
        assert "results" in data
        assert "total_images" in data
        assert "successful_analyses" in data
        assert "failed_analyses" in data
        assert "total_processing_time_seconds" in data
        
        assert data["total_images"] == 2
        assert data["successful_analyses"] == 2
        assert len(data["results"]) == 2
    
    def test_model_info_endpoint(self, test_client):
        """Test model information endpoint."""
        response = test_client.get("/api/v1/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "total_memory_usage_mb" in data
    
    def test_model_reload_endpoint(self, test_client):
        """Test model reload endpoint."""
        response = test_client.post("/api/v1/models/reload")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "message" in data
        assert "timestamp" in data
    
    def test_invalid_file_type(self, test_client):
        """Test handling of invalid file types."""
        # Create text file instead of image
        text_content = b"This is not an image"
        
        response = test_client.post(
            "/api/v1/analyze",
            files={"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
        )
        
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
    
    def test_batch_size_limit(self, test_client):
        """Test batch size limitation."""
        # Create too many files
        files = []
        for i in range(51):  # Exceeds limit of 50
            files.append(("file", (f"test{i}.jpg", io.BytesIO(b"fake"), "image/jpeg")))
        
        response = test_client.post("/api/v1/analyze/batch", files=files)
        assert response.status_code == 400
        assert "Batch size limited to 50 images" in response.json()["detail"]

class TestModelServer:
    """Test model server deployment utilities."""
    
    @patch('src.swellsight.api.deployment.WaveAnalysisPipeline')
    def test_model_server_initialization(self, mock_pipeline_class):
        """Test model server initialization."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.get_pipeline_status.return_value = {
            "components_initialized": True,
            "gpu_available": True
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        # Create model server
        server = ModelServer()
        success = server.initialize_models()
        
        assert success is True
        assert server.pipeline is not None
        assert server.health_status["status"] == "healthy"
    
    @patch('src.swellsight.api.deployment.WaveAnalysisPipeline')
    def test_model_server_health_status(self, mock_pipeline_class):
        """Test model server health status reporting."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.get_pipeline_status.return_value = {
            "components_initialized": True,
            "gpu_available": True
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        # Create and initialize server
        server = ModelServer()
        server.initialize_models()
        
        # Get health status
        health = server.get_health_status()
        
        assert "status" in health
        assert "uptime_seconds" in health
        assert "system_metrics" in health
        assert "pipeline_metrics" in health
    
    @patch('src.swellsight.api.deployment.WaveAnalysisPipeline')
    def test_model_server_reload(self, mock_pipeline_class):
        """Test model server reload functionality."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.get_pipeline_status.return_value = {
            "components_initialized": True,
            "gpu_available": True
        }
        mock_pipeline._initialize_components = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Create and initialize server
        server = ModelServer()
        server.initialize_models()
        
        # Test reload
        result = server.reload_models()
        
        assert result["status"] == "success"
        assert "message" in result
        assert "timestamp" in result
        mock_pipeline._initialize_components.assert_called_once()
    
    def test_model_server_cleanup(self):
        """Test model server cleanup."""
        server = ModelServer()
        server.pipeline = Mock()
        server.pipeline.cleanup = Mock()
        
        # Test cleanup
        server.cleanup()
        
        assert server.pipeline is None
        assert server.health_status["status"] == "shutdown"

class TestResultCaching:
    """Test result caching functionality."""
    
    @pytest.fixture
    def test_client_with_cache(self, mock_pipeline):
        """Create test client for caching tests."""
        app = create_app()
        app.state.pipeline = mock_pipeline
        return TestClient(app)
    
    def test_cache_hit(self, test_client_with_cache):
        """Test cache hit for identical images."""
        # Create test image
        image = Image.new('RGB', (640, 480), color='blue')
        img_bytes1 = io.BytesIO()
        image.save(img_bytes1, format='JPEG')
        img_bytes1.seek(0)
        
        img_bytes2 = io.BytesIO()
        image.save(img_bytes2, format='JPEG')
        img_bytes2.seek(0)
        
        # First request
        response1 = test_client_with_cache.post(
            "/api/v1/analyze",
            files={"file": ("test1.jpg", img_bytes1, "image/jpeg")}
        )
        assert response1.status_code == 200
        
        # Second request with same image should hit cache
        response2 = test_client_with_cache.post(
            "/api/v1/analyze",
            files={"file": ("test2.jpg", img_bytes2, "image/jpeg")}
        )
        assert response2.status_code == 200
        
        # Results should be identical
        assert response1.json() == response2.json()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])