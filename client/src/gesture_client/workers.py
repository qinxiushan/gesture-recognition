from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import httpx
import numpy as np
from PySide6.QtCore import QThread, Signal

from .api import RobotApi
from .config import ClientSettings
from .gate import GestureGate
from .mjpeg import MjpegParser
from .model_cache import ModelCache


@dataclass(frozen=True)
class VideoFrame:
    sequence: int
    received_at: float
    bgr: np.ndarray


class PipelineWorker(QThread):
    """Preview pulls the latest frame; inference and HTTP never block the GUI."""

    connection_changed = Signal(str)
    log_message = Signal(str)

    def __init__(self, settings: ClientSettings, model_cache: ModelCache | None = None):
        super().__init__()
        self.settings = settings
        self.model_cache = model_cache if model_cache is not None else ModelCache()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: VideoFrame | None = None
        self._result: dict | None = None
        self._busy = True
        self._pending = False
        self._received = self._inferred = self._bytes = 0
        self._started_at = time.monotonic()
        self._first_frame_s: float | None = None
        self._connection_ms = 0.0
        self._model_load_s = 0.0
        self._decode_ms = self._inference_ms = 0.0
        self._sample_at = self._started_at
        self._sample_counts = (0, 0, 0)
        self._rates = (0.0, 0.0, 0.0)

    def stop(self) -> None:
        self._stop_event.set()

    def snapshot(self, after_sequence: int = 0):
        """Called by a GUI timer, not one queued Qt signal per incoming frame."""
        now = time.monotonic()
        with self._lock:
            latest = self._latest
            new_frame = latest if latest and latest.sequence > after_sequence else None
            result = self._result
            elapsed = now - self._sample_at
            if elapsed >= 1.0:
                counts = (self._received, self._inferred, self._bytes)
                self._rates = tuple((n - p) / elapsed for n, p in zip(counts, self._sample_counts))
                self._sample_counts, self._sample_at = counts, now
            metrics = {
                'receive_fps': self._rates[0], 'inference_fps': self._rates[1],
                'mbps': self._rates[2] * 8 / 1_000_000,
                'received_frames': self._received, 'inferred_frames': self._inferred,
                'first_frame_s': self._first_frame_s, 'connection_ms': self._connection_ms,
                'model_load_s': self._model_load_s, 'decode_ms': self._decode_ms,
                'inference_ms': self._inference_ms,
                'frame_age_ms': (now - latest.received_at) * 1000 if latest else None,
                'arm_busy': self._busy or self._pending,
            }
            return new_frame, result, metrics

    def _read_stream(self, session_stop: threading.Event, failed: threading.Event) -> None:
        try:
            with httpx.stream('GET', f'{self.settings.base_url}/video/stream',
                             timeout=httpx.Timeout(3.0), trust_env=False) as response:
                response.raise_for_status()
                parser = MjpegParser(response.headers.get('content-type', ''))
                last_source_sequence = -1
                for chunk in response.iter_bytes():
                    if session_stop.is_set() or self._stop_event.is_set():
                        return
                    with self._lock:
                        self._bytes += len(chunk)
                    packets = parser.feed(chunk)
                    if not packets:
                        continue
                    # If a socket read contains old frames, decode only its newest.
                    packet = packets[-1]
                    if packet.sequence is not None:
                        if packet.sequence <= last_source_sequence:
                            continue
                        last_source_sequence = packet.sequence
                    started = time.perf_counter()
                    frame = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    now = time.monotonic()
                    with self._lock:
                        self._received += 1
                        self._decode_ms = (time.perf_counter() - started) * 1000
                        self._latest = VideoFrame(self._received, now, frame)
                        if self._first_frame_s is None:
                            self._first_frame_s = now - self._started_at
            if not session_stop.is_set() and not self._stop_event.is_set():
                raise RuntimeError('视频流已结束')
        except Exception as exc:
            if not session_stop.is_set() and not self._stop_event.is_set():
                self.log_message.emit(f'视频流中断: {exc}')
                failed.set()

    def _control(self, session_stop: threading.Event, failed: threading.Event,
                 actions: queue.Queue) -> None:
        api = RobotApi(self.settings.base_url)
        next_status = 0.0
        try:
            while not session_stop.is_set() and not self._stop_event.is_set():
                if time.monotonic() >= next_status:
                    status = api.status()
                    with self._lock:
                        self._busy = status.arm.busy
                    next_status = time.monotonic() + 0.5
                try:
                    observed_at, detection = actions.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if session_stop.is_set() or self._stop_event.is_set():
                        return
                    if time.monotonic() - observed_at > self.settings.max_frame_age_seconds:
                        self.log_message.emit('动作已过期，跳过')
                        continue
                    result = api.action(detection.label, detection.confidence,
                                        detection.x, detection.y)
                    with self._lock:
                        self._busy = result.accepted
                    self.log_message.emit(f'动作 {detection.label}: ' +
                        ('已接受' if result.accepted else f'已拒绝 ({result.reason})'))
                finally:
                    with self._lock:
                        self._pending = False
                next_status = 0.0
        except Exception as exc:
            if not session_stop.is_set() and not self._stop_event.is_set():
                self.log_message.emit(f'控制连接中断: {exc}')
                failed.set()
        finally:
            api.close()

    def run(self) -> None:
        self._started_at = time.monotonic()
        try:
            job, reused = self.model_cache.acquire(self.settings)
        except Exception as exc:
            self.log_message.emit(f'模型初始化失败: {exc}')
            return
        cached_ready = reused and job.ready.is_set()
        self.log_message.emit('复用模型缓存' if cached_ready else '后台加载模型，先连接视频')
        model_reported = False
        try:
            for attempt in range(1, 6):
                if self._stop_event.is_set():
                    break
                self.connection_changed.emit(f'连接中 ({attempt}/5)')
                api = RobotApi(self.settings.base_url)
                connected_at = time.monotonic()
                try:
                    health = api.health()
                    with self._lock:
                        self._connection_ms = (time.monotonic() - connected_at) * 1000
                    self.connection_changed.emit('已连接，模型加载中' if not job.ready.is_set() else '已连接')
                    self.log_message.emit(f'已连接服务端协议 {health.api_version}')
                except Exception as exc:
                    self.log_message.emit(f'连接失败: {exc}')
                    self._stop_event.wait(min(2 ** (attempt - 1), 8))
                    continue
                finally:
                    api.close()

                session_stop, failed = threading.Event(), threading.Event()
                actions: queue.Queue = queue.Queue(maxsize=1)
                gate = GestureGate(self.settings.confidence, self.settings.stable_frames,
                                   self.settings.cooldown_seconds)
                with self._lock:
                    self._latest = self._result = None
                    self._busy, self._pending = True, False
                reader = threading.Thread(target=self._read_stream, args=(session_stop, failed),
                                          name='video-reader', daemon=True)
                control = threading.Thread(target=self._control, args=(session_stop, failed, actions),
                                           name='robot-control', daemon=True)
                reader.start()
                control.start()
                last_sequence, next_inference = 0, 0.0
                try:
                    while not self._stop_event.is_set() and not failed.is_set():
                        if not job.ready.is_set():
                            self._stop_event.wait(0.02)
                            continue
                        if not model_reported:
                            with self._lock:
                                self._model_load_s = 0.0 if cached_ready else job.elapsed_s
                            if job.error is not None:
                                self.log_message.emit(f'模型加载失败，视频仍可预览: {job.error}')
                                self.connection_changed.emit('已连接，识别不可用')
                            else:
                                self.log_message.emit('缓存模型已就绪' if cached_ready else
                                                      f'模型就绪 ({job.elapsed_s:.2f} 秒)')
                                self.connection_changed.emit('已连接，识别就绪')
                            model_reported = True
                        if job.error is not None:
                            self._stop_event.wait(0.05)
                            continue
                        now = time.monotonic()
                        with self._lock:
                            frame = self._latest
                            busy = self._busy or self._pending
                        if (frame is None or frame.sequence == last_sequence or now < next_inference or
                                now - frame.received_at > self.settings.max_frame_age_seconds):
                            self._stop_event.wait(0.01)
                            continue
                        last_sequence = frame.sequence
                        next_inference = now + 1 / max(1, self.settings.inference_fps)
                        detection = job.detector.detect(frame.bgr, render=False)
                        finished_at = time.monotonic()
                        with self._lock:
                            self._inferred += 1
                            self._inference_ms = detection.elapsed_ms
                        if self._stop_event.is_set() or failed.is_set():
                            break
                        if finished_at - frame.received_at > self.settings.max_frame_age_seconds:
                            gate.reset()
                            continue
                        result = {'label': detection.label or 'none',
                                  'confidence': detection.confidence, 'elapsed_ms': detection.elapsed_ms,
                                  'boxes': detection.boxes, 'observed_at': frame.received_at,
                                  'finished_at': finished_at, 'shape': frame.bgr.shape,
                                  'sequence': frame.sequence}
                        with self._lock:
                            self._result = result
                            busy = self._busy or self._pending
                        if gate.observe(detection.label, detection.confidence, busy):
                            with self._lock:
                                self._pending = True
                            try:
                                actions.put_nowait((frame.received_at, detection))
                            except queue.Full:
                                with self._lock:
                                    self._pending = False
                finally:
                    session_stop.set()
                    reader.join(timeout=4)
                    control.join(timeout=4)
                    if reader.is_alive() or control.is_alive():
                        raise RuntimeError('连接线程未能及时退出，请重新启动客户端')
                    with self._lock:
                        self._latest = self._result = None
                if self._stop_event.is_set():
                    break
                if attempt < 5:
                    delay = min(2 ** (attempt - 1), 8)
                    self.connection_changed.emit(f'重连中 ({attempt}/5)')
                    self._stop_event.wait(delay)
            else:
                self.connection_changed.emit('重连失败')
        except Exception as exc:
            self.log_message.emit(f'运行失败: {exc}')
        finally:
            self._stop_event.set()
            self.connection_changed.emit('已停止')
