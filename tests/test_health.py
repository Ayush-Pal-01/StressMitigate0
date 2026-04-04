"""
test_health.py — Tests for the health check endpoint.
"""
import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Health endpoint should return 200 with model and system info."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert "text" in data["models"]
    assert "face" in data["models"]
    assert "voice" in data["models"]
    assert "database" in data
    assert "system" in data
    assert "memory_rss_mb" in data["system"]
    assert "python_version" in data["system"]


@pytest.mark.asyncio
async def test_health_model_backends(client):
    """Health should report model backend type (onnx, tflite, keras, etc)."""
    response = await client.get("/health")
    data = response.json()

    for model_name in ["text", "face", "voice"]:
        model_info = data["models"][model_name]
        assert "loaded" in model_info
        assert "backend" in model_info
        assert isinstance(model_info["loaded"], bool)
        assert model_info["backend"] in ["none", "onnx_int8", "tensorflow", "tflite", "keras"]
