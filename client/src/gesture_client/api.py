from __future__ import annotations

from datetime import datetime, timezone
import uuid

import httpx
from gesture_shared import ActionRequest, ActionResponse, Gesture, HealthResponse, StatusResponse


class RobotApi:
    def __init__(self, base_url: str, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        # Robot traffic is local-LAN traffic and must not be routed through a
        # desktop HTTP proxy configured for internet access.
        self.client = httpx.Client(timeout=timeout, trust_env=False)

    def health(self) -> HealthResponse:
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return HealthResponse.model_validate(response.json())

    def status(self) -> StatusResponse:
        response = self.client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return StatusResponse.model_validate(response.json())

    def action(self, gesture: str, confidence: float, x: float, y: float) -> ActionResponse:
        payload = ActionRequest(request_id=str(uuid.uuid4()), gesture=Gesture(gesture),
                                confidence=confidence, x=x, y=y,
                                client_time=datetime.now(timezone.utc))
        response = self.client.post(f"{self.base_url}/actions",
                                    json=payload.model_dump(mode="json"))
        response.raise_for_status()
        return ActionResponse.model_validate(response.json())

    def pose(self, pose: str) -> ActionResponse:
        response = self.client.post(f"{self.base_url}/arm/pose", json={"pose": pose})
        response.raise_for_status()
        return ActionResponse.model_validate(response.json())

    def close(self) -> None:
        self.client.close()
