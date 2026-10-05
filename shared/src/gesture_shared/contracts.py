from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = "v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Gesture(str, Enum):
    FIVE = "five"
    OK = "ok"
    SIX = "six"
    LIKE = "like"
    YES = "yes"


class Pose(str, Enum):
    RECOGNITION = "recognition"
    SAFE = "safe"


class ActionState(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraStatus(StrictModel):
    available: bool
    device: str
    last_error: str | None = None


class ArmStatus(StrictModel):
    available: bool
    busy: bool
    current_action: str | None = None
    last_action: str | None = None
    last_error: str | None = None


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    api_version: str = API_VERSION
    camera: CameraStatus
    arm: ArmStatus


class StatusResponse(StrictModel):
    arm: ArmStatus


class ActionRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    gesture: Gesture
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = 0.0
    y: float = 0.0
    client_time: datetime = Field(default_factory=utc_now)


class ActionResponse(StrictModel):
    accepted: bool
    request_id: str
    action_id: str | None = None
    state: ActionState
    reason: str | None = None


class PoseRequest(StrictModel):
    pose: Pose

