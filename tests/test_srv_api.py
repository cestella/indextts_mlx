"""Integration tests for the FastAPI endpoints."""

import time

import pytest

from indextts_mlx.srv.app import create_app
from indextts_mlx.srv.config import SrvConfig


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    config = SrvConfig(heartbeat_timeout_s=300.0)
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


def test_submit_and_get(client):
    r = client.post("/jobs", json={
        "model_type": "mock",
        "application_id": "test",
        "priority": 10,
        "payload": {"duration": 0.01},
    })
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == job_id
    assert data["model_type"] == "mock"


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "queue_depth" in data
    assert "uptime_s" in data


def test_queue_endpoint(client):
    # Submit a job first
    client.post("/jobs", json={
        "model_type": "mock",
        "application_id": "test",
        "priority": 10,
        "payload": {},
    })

    r = client.get("/queue")
    assert r.status_code == 200
    data = r.json()
    assert "active" in data
    assert "queued" in data
    assert "recent" in data


def test_cancel_endpoint(client):
    r = client.post("/jobs", json={
        "model_type": "mock",
        "application_id": "test",
        "priority": 10,
        "payload": {"duration": 10.0},  # long duration so it stays queued
    })
    job_id = r.json()["job_id"]

    # Cancel quickly before worker picks it up
    r = client.delete(f"/jobs/{job_id}")
    # It might already be running, so accept either outcome
    assert r.status_code in (200, 404)


def test_job_completes(client):
    r = client.post("/jobs", json={
        "model_type": "mock",
        "application_id": "test",
        "priority": 10,
        "payload": {"duration": 0.01},
    })
    job_id = r.json()["job_id"]

    # Poll until done
    for _ in range(50):
        r = client.get(f"/jobs/{job_id}")
        if r.json()["status"] in ("done", "failed"):
            break
        time.sleep(0.05)

    assert r.json()["status"] == "done"


def test_unknown_job_404(client):
    r = client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
