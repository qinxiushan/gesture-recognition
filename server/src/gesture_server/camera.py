import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def get_jpeg(self, timeout: float = 2.0) -> Optional[bytes]: ...

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @property
    @abstractmethod
    def last_error(self) -> Optional[str]: ...

    @property
    @abstractmethod
    def device(self) -> str: ...


class OpenCVCamera(CameraSource):
    def __init__(self, source: Union[int, str], width: int, height: int, fps: int, quality: int):
        self.source, self.width, self.height = source, width, height
        self.fps, self.quality = fps, quality
        self._capture = None  # type: Optional[cv2.VideoCapture]
        self._jpeg = None  # type: Optional[bytes]
        self._error = None  # type: Optional[str]
        self._running = False
        self._condition = threading.Condition()
        self._thread = None  # type: Optional[threading.Thread]

    @property
    def device(self) -> str:
        return str(self.source)

    @property
    def available(self) -> bool:
        return self._running and self._jpeg is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def start(self) -> None:
        if self._running:
            return
        self._capture = cv2.VideoCapture(self.source)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"cannot open camera {self.source}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            started = time.monotonic()
            assert self._capture is not None
            ok, frame = self._capture.read()
            if not ok:
                self._error = "camera read failed"
                time.sleep(0.2)
                continue
            encoded, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if encoded:
                with self._condition:
                    self._jpeg = data.tobytes()
                    self._error = None
                    self._condition.notify_all()
            time.sleep(max(0.0, interval - (time.monotonic() - started)))

    def get_jpeg(self, timeout: float = 2.0) -> Optional[bytes]:
        with self._condition:
            if self._jpeg is None:
                self._condition.wait(timeout)
            return self._jpeg

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._capture:
            self._capture.release()
        self._capture = None


class MockCamera(CameraSource):
    def __init__(self, width: int = 640, height: int = 480, quality: int = 80):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, "GESTURE SERVER - MOCK CAMERA", (30, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("failed to create mock camera frame")
        self._jpeg = data.tobytes()
        self._running = False

    device = "mock"
    last_error = None

    @property
    def available(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def get_jpeg(self, timeout: float = 2.0) -> Optional[bytes]:
        return self._jpeg if self._running else None

