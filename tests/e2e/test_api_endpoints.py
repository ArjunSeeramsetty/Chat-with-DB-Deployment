"""
End-to-end tests for API endpoints
Sprint 6: PyTest matrix and GitHub Actions CI
"""
import pytest
import asyncio
import httpx
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.main import app
from backend.config import get_settings

class TestAPIEndpoints:
    """Test API endpoints end-to-end"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    @pytest.fixture
    def settings(self):
        """Get application settings"""
        return get_settings()
    
    @pytest.mark.e2e
    @pytest.mark.sprint1
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
    
    @pytest.mark.e2e
    @pytest.mark.sprint1
    def test_llm_models_endpoint(self, client):
        """Test LLM models endpoint"""
        response = client.get("/api/v1/llm/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "current_model" in data
        assert len(data["models"]) > 0
    
    @pytest.mark.e2e
    @pytest.mark.sprint1
    def test_gpu_status_endpoint(self, client):
        """Test GPU status endpoint"""
        response = client.get("/api/v1/llm/gpu-status")
        assert response.status_code == 200
        
        data = response.json()
        assert "gpu_available" in data
        assert "gpu_enabled" in data
    
    @pytest.mark.e2e
    @pytest.mark.sprint2
    def test_ask_endpoint_valid_query(self, client):
        """Test ask endpoint with valid query"""
        request_data = {
            "question": "What is the energy consumption?",
            "user_id": "test_user",
            "processing_mode": "balanced",
            "clarification_attempt_count": 0
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        assert response.status_code in [200, 422]  # 422 for clarification needed
        
        data = response.json()
        assert "success" in data
        assert "confidence" in data
        assert "processing_time" in data
    
    @pytest.mark.e2e
    @pytest.mark.sprint2
    @pytest.mark.clarification
    def test_ask_endpoint_ambiguous_query(self, client):
        """Test ask endpoint with ambiguous query that needs clarification"""
        request_data = {
            "question": "What is the energy?",
            "user_id": "test_user",
            "processing_mode": "balanced",
            "clarification_attempt_count": 0
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        assert response.status_code == 422  # Should request clarification
        
        data = response.json()
        assert data["success"] is False
        assert "clarification_question" in data
        assert data["clarification_needed"] is True
    
    @pytest.mark.e2e
    @pytest.mark.sprint3
    def test_schema_endpoint(self, client):
        """Test schema endpoint"""
        response = client.get("/api/v1/schema")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "schema" in data
        assert "timestamp" in data
        assert data["success"] is True
    
    @pytest.mark.e2e
    @pytest.mark.sprint5
    @pytest.mark.validation
    def test_validate_sql_endpoint(self, client):
        """Test SQL validation endpoint"""
        request_data = {
            "sql": "SELECT StateName FROM FactStateDailyEnergy"
        }
        
        response = client.post("/api/v1/validate-sql", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "is_valid" in data
        assert "confidence" in data
        assert data["is_valid"] is True
    
    @pytest.mark.e2e
    @pytest.mark.sprint5
    @pytest.mark.security
    def test_validate_sql_endpoint_dangerous_query(self, client):
        """Test SQL validation endpoint with dangerous query"""
        request_data = {
            "sql": "DROP TABLE FactStateDailyEnergy"
        }
        
        response = client.post("/api/v1/validate-sql", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "is_valid" in data
        assert data["is_valid"] is False
        assert "DROP" in str(data.get("errors", []))
    
    @pytest.mark.e2e
    @pytest.mark.sprint1
    def test_cache_invalidate_endpoint(self, client):
        """Test cache invalidation endpoint"""
        response = client.post("/api/v1/cache/invalidate")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "timestamp" in data
        assert data["success"] is True
    
    @pytest.mark.e2e
    @pytest.mark.sprint1
    def test_config_reload_endpoint(self, client):
        """Test config reload endpoint"""
        response = client.get("/api/v1/config/reload")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "timestamp" in data
        assert data["success"] is True
    
    @pytest.mark.e2e
    @pytest.mark.sprint3
    def test_entities_reload_endpoint(self, client):
        """Test entities reload endpoint"""
        response = client.post("/api/v1/entities/reload")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "timestamp" in data
        assert data["success"] is True
    
    @pytest.mark.e2e
    @pytest.mark.sprint2
    def test_feedback_endpoint(self, client):
        """Test feedback endpoint"""
        request_data = {
            "original_query": "What is the energy consumption?",
            "generated_sql": "SELECT StateName, SUM(EnergyMet) FROM FactStateDailyEnergy",
            "feedback_text": "Good query",
            "is_correct": True,
            "user_id": "test_user",
            "session_id": "test_session",
            "regenerate": False
        }
        
        response = client.post("/api/v1/feedback", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
    
    @pytest.mark.e2e
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert "features" in data

class TestAPIErrorHandling:
    """Test API error handling"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    @pytest.mark.e2e
    def test_ask_endpoint_empty_question(self, client):
        """Test ask endpoint with empty question"""
        request_data = {
            "question": "",
            "user_id": "test_user",
            "processing_mode": "balanced",
            "clarification_attempt_count": 0
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "empty" in data["detail"].lower()
    
    @pytest.mark.e2e
    def test_ask_endpoint_long_question(self, client):
        """Test ask endpoint with very long question"""
        long_question = "What is the energy consumption? " * 50  # Very long question
        request_data = {
            "question": long_question,
            "user_id": "test_user",
            "processing_mode": "balanced",
            "clarification_attempt_count": 0
        }
        
        response = client.post("/api/v1/ask", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "long" in data["detail"].lower()
    
    @pytest.mark.e2e
    def test_validate_sql_endpoint_empty_sql(self, client):
        """Test SQL validation endpoint with empty SQL"""
        request_data = {
            "sql": ""
        }
        
        response = client.post("/api/v1/validate-sql", json=request_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "required" in data["detail"].lower() 