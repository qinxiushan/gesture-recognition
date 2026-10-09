import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraFrame(NamedTuple):
    sequence: int
    captured_at: float
    jpeg: bytes


class CameraSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def get_frame(self, after_sequence: int = 0,
                  timeout: float = 2.0) -> Optional[CameraFrame]: ...

    def get_jpeg(self, timeout: float = 2.0) -> Optional[bytes]:
        frame = self.get_frame(timeout=timeout)
        return frame.jpeg if frame else None

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @property
    @abstractmethod
    def last_error(self) -> Optional[str]: ...

    @property
    @abstractmethod
    def device(self) -> str: ...


class BufferedCamera(CameraSource):
    """One latest frame; every subscriber has its own sequence cursor."""

    def __init__(self):
        self._frame = None  # type: Optional[CameraFrame]
        self._sequence = 0
        self._updated_at = 0.0
        self._error = None  # type: Optional[str]
        self._running = False
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = None  # type: Optional[threading.Thread]

    @property
    def available(self) -> bool:
        with self._condition:
            return (self._running and self._frame is not None and
                    time.monotonic() - self._updated_at < 2.0)

    @property
    def last_error(self) -> Optional[str]:
        if self._running and self._frame is not None and time.monotonic() - self._updated_at >= 2.0:
            return "camera frame timeout"
        return self._error

    def _publish(self, jpeg: bytes, captured_at: float) -> None:
        with self._condition:
            self._sequence += 1
            self._frame = CameraFrame(self._sequence, captured_at, jpeg)
            self._updated_at = time.monotonic()
            self._error = None
            self._condition.notify_all()

    def get_frame(self, after_sequence: int = 0,
                  timeout: float = 2.0) -> Optional[CameraFrame]:
        with self._condition:
            self._condition.wait_for(
                lambda: not self._running or
                (self._frame is not None and self._frame.sequence > after_sequence),
                timeout=timeout)
            if (self._running and self._frame is not None and
                    self._frame.sequence > after_sequence and
                    time.monotonic() - self._updated_at < 2.0):
                return self._frame
            return None

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._running = False
            self._frame = None
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=3)


class OpenCVCamera(BufferedCamera):
    def __init__(self, source: Union[int, str], width: int, height: int, fps: int, quality: int):
        super().__init__()
        self.source, self.width, self.height = source, width, height
        self.fps, self.quality = fps, quality
        self._capture = None  # type: Optional[cv2.VideoCapture]

    @property
    def device(self) -> str:
        return str(self.source)

    def start(self) -> None:
        if self._running:
            return
        self._capture = cv2.VideoCapture(self.source,
            cv2.CAP_DSHOW if os.name == "nt" and isinstance(self.source, int) else
            cv2.CAP_V4L2 if os.name != "nt" and
            (isinstance(self.source, int) or str(self.source).startswith("/dev/video")) else cv2.CAP_ANY)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError("cannot open camera {}".format(self.source))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        next_frame_at = time.monotonic()
        while not self._stop_event.is_set():
            # Drain the driver's native frame rate, even if JPEG output is slower.
            # Sleeping between reads can accumulate old frames in the USB driver.
            ok, frame = self._capture.read()
            captured_at = time.time()
            if not ok:
                with self._condition:
                    self._error = "camera read failed"
                    self._frame = None
                self._stop_event.wait(0.2)
                continue
            now = time.monotonic()
            if now + 0.002 < next_frame_at:
                continue
            next_frame_at = max(next_frame_at + interval, now)
            encoded, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if encoded:
                self._publish(data.tobytes(), captured_at)

    def stop(self) -> None:
        super().stop()
        if self._capture and (self._thread is None or not self._thread.is_alive()):
            self._capture.release()
            self._capture = None


class MockCamera(BufferedCamera):
    device = "mock"

    def __init__(self, width: int = 640, height: int = 480, quality: int = 80, fps: int = 15):
        super().__init__()
        self.width, self.height, self.quality, self.fps = width, height, quality, fps

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="mock-camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            x = 20 + (self._sequence * 9) % max(1, self.width - 60)
            cv2.circle(frame, (x, self.height // 3), 18, (0, 200, 255), -1)
            cv2.putText(frame, "MOCK FRAME {}".format(self._sequence + 1),
                        (20, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            encoded, data = cv2.imencode(".jpg", frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if encoded:
                self._publish(data.tobytes(), time.time())
            self._stop_event.wait(max(0.0, 1.0 / self.fps - (time.monotonic() - started)))
