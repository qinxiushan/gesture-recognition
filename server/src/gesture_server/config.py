from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GESTURE_SERVER_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = Field(default=8896, ge=1, le=65535)
    camera: str = "0"
    width: int = Field(default=640, ge=160, le=4096)
    height: int = Field(default=480, ge=120, le=2160)
    fps: int = Field(default=15, ge=1, le=60)
    jpeg_quality: int = Field(default=80, ge=20, le=100)
    dry_run: bool = False
    mock_camera: bool = False
    start_pose: str = "recognition"

    @field_validator("start_pose")
    @classmethod
    def valid_pose(cls, value: str) -> str:
        if value not in {"recognition", "safe"}:
            raise ValueError("start_pose must be recognition or safe")
        return value

    @property
    def camera_source(self) -> int | str:
        return int(self.camera) if self.camera.isdecimal() else self.camera

