import time

from fastapi.testclient import TestClient

from gesture_server.app import Runtime, create_app
from gesture_server.arm import ArmController, DryRunArm
from gesture_server.camera import MockCamera
from gesture_server.config import ServerSettings


def make_client():
    runtime = Runtime(MockCamera(320, 240), ArmController(DryRunArm(0.01)))
    settings = ServerSettings(dry_run=True, mock_camera=True, width=320, height=240)
    return TestClient(create_app(settings, runtime)), runtime


def test_health_status_and_action_flow():
    client, runtime = make_client()
    with client:
        time.sleep(0.08)
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["camera"]["available"] is True
        payload = {"request_id": "req-1", "gesture": "ok", "confidence": 0.91,
                   "x": 12, "y": 34, "client_time": "2026-01-01T00:00:00Z"}
        response = client.post("/api/v1/actions", json=payload)
        assert response.status_code == 200
        assert response.json()["accepted"] is True
        duplicate = client.post("/api/v1/actions", json=payload)
        assert duplicate.json()["accepted"] is False
        assert duplicate.json()["reason"] == "duplicate request_id"
        assert client.get("/api/v1/status").status_code == 200
        jpeg = runtime.camera.get_jpeg()
        assert jpeg and jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")


def test_invalid_action_returns_validation_error():
    client, _ = make_client()
    with client:
        response = client.post("/api/v1/actions", json={
            "request_id": "bad", "gesture": "unknown", "confidence": 2,
        })
        assert response.status_code == 422

