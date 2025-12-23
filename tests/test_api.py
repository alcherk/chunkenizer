"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "sqlite" in data
    assert "qdrant" in data
    assert "embedding_model" in data


def test_list_documents_empty():
    """Test listing documents when none exist."""
    response = client.get("/api/documents")
    
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)

