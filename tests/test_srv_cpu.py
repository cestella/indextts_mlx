"""Tests for CPU-only endpoints."""

import time

import pytest

from indextts_mlx.srv.app import create_app
from indextts_mlx.srv.config import SrvConfig


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    config = SrvConfig()
    models_config = {
        "backends": {
            "mock": {
                "default": "default",
                "models": {"default": {}},
            },
        },
    }
    app = create_app(config=config, models_config=models_config)
    with TestClient(app) as c:
        yield c


def test_normalize_endpoint(client):
    r = client.post("/cpu/normalize", json={"text": "The price is $123.", "language": "en"})
    assert r.status_code == 200
    data = r.json()
    assert "text" in data
    # NeMo should have expanded "$123" into words
    assert "$" not in data["text"]
    assert "123" not in data["text"]


def test_segment_endpoint(client):
    long_text = "This is sentence one. This is sentence two. This is sentence three. "
    long_text += "Another sentence here. And yet another one. Final sentence of the batch."

    r = client.post("/cpu/segment", json={
        "text": long_text,
        "language": "english",
        "max_chars": 50,
    })
    assert r.status_code == 200
    data = r.json()
    assert "segments" in data
    assert isinstance(data["segments"], list)
    assert len(data["segments"]) > 0


def test_cpu_concurrent_with_gpu(client):
    """CPU request works while a GPU job is running."""
    # Submit a slow GPU job
    r = client.post("/jobs", json={
        "model_type": "mock",
        "application_id": "test",
        "priority": 10,
        "payload": {"duration": 1.0},
    })
    assert r.status_code == 200

    # CPU segment request should work immediately
    r = client.post("/cpu/segment", json={
        "text": "Hello. World.",
        "language": "english",
    })
    assert r.status_code == 200
    assert len(r.json()["segments"]) > 0
