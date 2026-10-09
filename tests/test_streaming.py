"""Real loopback HTTP + Qt pipeline tests, with injected slow compute/control."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from PySide6.QtWidgets import QApplication

from gesture_client.api import RobotApi
from gesture_client.config import ClientSettings
from gesture_client.inference import Detection
from gesture_client.mjpeg import MjpegParser
from gesture_client.model_cache import ModelCache
from gesture_client.workers import PipelineWorker
from gesture_server.app import Runtime, create_app
from gesture_server.arm import ArmController, DryRunArm
from gesture_server.camera import MockCamera
from gesture_server.config import ServerSettings


@pytest.fixture(scope='module')
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def live_server():
    camera = MockCamera(320, 240, fps=15)
    hardware = DryRunArm(0.02)
    runtime = Runtime(camera, ArmController(hardware))
    app = create_app(ServerSettings(dry_run=True, mock_camera=True), runtime)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started
    try:
        yield SimpleNamespace(port=port, camera=camera, hardware=hardware)
    finally:
        server.should_exit = True
        thread.join(timeout=6)
        sock.close()
        assert not thread.is_alive()


def test_each_http_subscriber_gets_new_frames_without_duplicate_flood(live_server):
    def collect():
        started = time.monotonic()
        sequences = []
        with httpx.stream('GET', f'http://127.0.0.1:{live_server.port}/api/v1/video/stream',
                          trust_env=False, timeout=3) as response:
            parser = MjpegParser(response.headers['content-type'])
            for chunk in response.iter_bytes():
                sequences.extend(p.sequence for p in parser.feed(chunk))
                if time.monotonic() - started >= 1.2:
                    break
        return sequences
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: collect(), range(2)))
    for sequences in results:
        assert 10 <= len(sequences) <= 23
        assert all(b > a for a, b in zip(sequences, sequences[1:]))


def test_camera_timeout_never_replays_previous_frame():
    camera = MockCamera(160, 120, fps=5)
    camera.start()
    try:
        first = camera.get_frame(timeout=1)
        assert first is not None
        assert camera.get_frame(first.sequence, timeout=0.01) is None
        second = camera.get_frame(first.sequence, timeout=1)
        assert second.sequence > first.sequence
        assert second.jpeg != first.jpeg
    finally:
        camera.stop()
    assert camera.get_frame(timeout=0) is None


class SlowDetector:
    def __init__(self, *args):
        time.sleep(1.0)

    def detect(self, bgr, render=False):
        started = time.monotonic()
        time.sleep(0.30)
        return Detection('five', 0.99, 50, 50, (time.monotonic() - started) * 1000,
                         bgr, [(10, 10, 100, 100, 0.99, 'five')])


def test_preview_does_not_wait_for_model_inference_or_slow_http(live_server, qt_app, monkeypatch):
    import gesture_client.workers as workers
    class SlowApi(RobotApi):
        def status(self):
            time.sleep(0.7)
            return super().status()

        def action(self, *args):
            time.sleep(0.7)
            return super().action(*args)
    monkeypatch.setattr(workers, 'RobotApi', SlowApi)
    cache = ModelCache(SlowDetector)
    settings = ClientSettings(host='127.0.0.1', port=live_server.port, stable_frames=1,
                              cooldown_seconds=0, max_frame_age_seconds=2)
    worker = PipelineWorker(settings, cache)
    sequences, received_at, last = [], [], 0
    started = time.monotonic()
    worker.start()
    try:
        while time.monotonic() - started < 4.0:
            qt_app.processEvents()
            frame, _, metrics = worker.snapshot(last)
            if frame is not None:
                last = frame.sequence
                sequences.append(last)
                received_at.append(time.monotonic() - started)
            time.sleep(0.01)
    finally:
        worker.stop()
        assert worker.wait(8000)
        qt_app.processEvents()
    assert received_at[0] < 0.8  # model deliberately takes 1 second
    assert len(sequences) >= 35  # preview remains much faster than 300 ms inference
    assert metrics['inferred_frames'] >= 5
    assert metrics['inferred_frames'] < len(sequences) / 2
    assert any(call == 'action:five' for call in live_server.hardware.calls)
    job, reused = cache.acquire(settings)
    assert reused and job.ready.is_set()


def test_reconnect_and_stop_clear_old_frames_and_keep_cached_model(live_server, qt_app):
    cache = ModelCache(lambda *args: SimpleNamespace(
        detect=lambda bgr, render=False: Detection(None, 0, 0, 0, 1, bgr)))
    worker = PipelineWorker(ClientSettings(host='127.0.0.1', port=live_server.port), cache)
    read_stream = worker._read_stream
    attempts = []
    def fail_once(session_stop, failed):
        attempts.append(1)
        if len(attempts) == 1:
            failed.set()
        else:
            read_stream(session_stop, failed)
    worker._read_stream = fail_once
    worker.start()
    deadline = time.monotonic() + 6
    try:
        while time.monotonic() < deadline:
            qt_app.processEvents()
            frame, _, _ = worker.snapshot()
            if frame is not None:
                break
            time.sleep(0.02)
        assert frame is not None and len(attempts) == 2
    finally:
        worker.stop()
        assert worker.wait(8000)
    assert worker.snapshot()[0] is None


def test_expired_inference_cannot_send_actions(live_server, qt_app):
    class ExpiredDetector:
        def __init__(self, *args):
            pass

        def detect(self, bgr, render=False):
            time.sleep(0.25)
            return Detection('five', 1.0, 0, 0, 250, bgr)
    worker = PipelineWorker(ClientSettings(host='127.0.0.1', port=live_server.port,
        stable_frames=1, max_frame_age_seconds=0.10), ModelCache(ExpiredDetector))
    worker.start()
    try:
        deadline = time.monotonic() + 1.8
        while time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.01)
        _, result, metrics = worker.snapshot()
        assert metrics['received_frames'] >= 10 and metrics['inferred_frames'] >= 3
        assert result is None
        assert not any(call.startswith('action:') for call in live_server.hardware.calls)
    finally:
        worker.stop()
        assert worker.wait(8000)


def test_model_failure_keeps_video_available(live_server, qt_app):
    def fail_model(*args):
        raise RuntimeError('missing test model')
    worker = PipelineWorker(ClientSettings(host='127.0.0.1', port=live_server.port),
                            ModelCache(fail_model))
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            qt_app.processEvents()
            frame, result, metrics = worker.snapshot()
            if metrics['received_frames'] >= 10:
                break
            time.sleep(0.01)
        assert frame is not None and result is None
        assert metrics['received_frames'] >= 10 and metrics['inferred_frames'] == 0
    finally:
        worker.stop()
        assert worker.wait(8000)
