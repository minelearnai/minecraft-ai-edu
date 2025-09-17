import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Minecraft AI Edu Backend"
    assert data["status"] == "running"

def test_health_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "backend"

def test_chat_endpoint():
    # Test basic chat functionality
    chat_data = {
        "user_id": "test_user",
        "message": "Co to jest matematyka?"
    }
    
    response = client.post("/chat", json=chat_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "response" in data
    assert data["user_id"] == "test_user"
    assert isinstance(data["sources"], list)

def test_knowledge_status_endpoint():
    response = client.get("/knowledge/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "document_count" in data or "message" in data