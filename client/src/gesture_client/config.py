from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ClientSettings:
    host: str = "192.168.1.11"
    port: int = 8896
    weights_path: str = "models/gesture_yolov5s_final.pt"
    yolo_source: str = "example/Dofbot_YOLO_Qt_Integrated_20260924_v3_recognition_pose/yolov5"
    confidence: float = 0.60
    stable_frames: int = 4
    cooldown_seconds: float = 3.5

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/api/v1"

    @classmethod
    def load(cls, path: Path) -> "ClientSettings":
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            allowed = cls.__dataclass_fields__.keys()
            return cls(**{key: value for key, value in values.items() if key in allowed})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

