from __future__ import annotations

import threading
import time

import cv2
import httpx
import numpy as np
from PySide6.QtCore import QThread, Signal

from .api import RobotApi
from .config import ClientSettings
from .gate import GestureGate
from .inference import YoloV5Detector


class PipelineWorker(QThread):
    frame_ready = Signal(object)
    detection_ready = Signal(dict)
    connection_changed = Signal(str)
    log_message = Signal(str)

    def __init__(self, settings: ClientSettings):
        super().__init__()
        self.settings = settings
        self._stop_event = threading.Event()
        self._latest: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stream_failed = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _read_stream(self) -> None:
        url = f"{self.settings.base_url}/video/stream"
        try:
            with httpx.stream("GET", url, timeout=httpx.Timeout(5, read=5),
                              trust_env=False) as response:
                response.raise_for_status()
                buffer = bytearray()
                for chunk in response.iter_bytes():
                    if self._stop_event.is_set():
                        return
                    buffer.extend(chunk)
                    while True:
                        start, end = buffer.find(b"\xff\xd8"), buffer.find(b"\xff\xd9")
                        if start < 0 or end < start:
                            break
                        jpeg = bytes(buffer[start:end + 2]); del buffer[:end + 2]
                        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._frame_lock:
                                self._latest = frame
        except Exception as exc:
            if not self._stop_event.is_set():
                self.log_message.emit(f"视频流中断: {exc}")
                self._stream_failed.set()

    def _take_latest(self) -> np.ndarray | None:
        with self._frame_lock:
            frame, self._latest = self._latest, None
            return frame

    def run(self) -> None:
        api: RobotApi | None = None
        gate = GestureGate(self.settings.confidence, self.settings.stable_frames,
                           self.settings.cooldown_seconds)
        try:
            self.connection_changed.emit("正在加载模型")
            detector = YoloV5Detector(self.settings.weights_path, self.settings.yolo_source,
                                      self.settings.confidence)
            self.log_message.emit("模型加载完成")
            for attempt in range(1, 6):
                if self._stop_event.is_set():
                    return
                api = RobotApi(self.settings.base_url)
                try:
                    health = api.health()
                    self.connection_changed.emit("已连接")
                    self.log_message.emit(f"已连接服务端协议 {health.api_version}")
                except Exception as exc:
                    self.connection_changed.emit(f"重连中 ({attempt}/5)")
                    self.log_message.emit(f"连接失败: {exc}")
                    api.close(); api = None
                    if attempt == 5:
                        return
                    self._stop_event.wait(min(2 ** (attempt - 1), 8))
                    continue

                self._stream_failed.clear()
                gate.reset()
                self._reader = threading.Thread(target=self._read_stream, daemon=True)
                self._reader.start()
                last_status = 0.0
                arm_busy = False
                while not self._stop_event.is_set() and not self._stream_failed.is_set():
                    frame = self._take_latest()
                    if frame is None:
                        self._stop_event.wait(0.01)
                        continue
                    detection = detector.detect(frame)
                    if time.monotonic() - last_status > 0.5:
                        try:
                            arm_busy = api.status().arm.busy
                        except Exception as exc:
                            self.log_message.emit(f"状态查询失败: {exc}")
                            self._stream_failed.set()
                            break
                        last_status = time.monotonic()
                    if gate.observe(detection.label, detection.confidence, arm_busy):
                        try:
                            result = api.action(detection.label or "", detection.confidence,
                                                detection.x, detection.y)
                            arm_busy = result.accepted
                            self.log_message.emit(f"动作 {detection.label}: " +
                                                  ("已接受" if result.accepted else f"已拒绝 ({result.reason})"))
                        except Exception as exc:
                            self.log_message.emit(f"动作发送失败: {exc}")
                            self._stream_failed.set()
                            break
                    self.frame_ready.emit(detection.rendered)
                    self.detection_ready.emit({"label": detection.label or "none",
                                               "confidence": detection.confidence,
                                               "elapsed_ms": detection.elapsed_ms,
                                               "arm_busy": arm_busy})

                api.close(); api = None
                if self._reader:
                    self._reader.join(timeout=6)
                    self._reader = None
                if self._stop_event.is_set():
                    return
                if attempt < 5:
                    delay = min(2 ** (attempt - 1), 8)
                    self.connection_changed.emit(f"重连中 ({attempt}/5)")
                    self.log_message.emit(f"连接已断开，{delay} 秒后重试")
                    self._stop_event.wait(delay)
                else:
                    self.connection_changed.emit("重连失败")
        except Exception as exc:
            self.log_message.emit(f"运行失败: {exc}")
        finally:
            self._stop_event.set()
            if self._reader: self._reader.join(timeout=2)
            if api: api.close()
            self.connection_changed.emit("已停止")
