"""Exercise real HTTP, real YOLO and actual Qt rendering; the robot is always mocked."""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / name / 'src') for name in ('client', 'server', 'shared')]
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import uvicorn
from PySide6.QtWidgets import QApplication
from gesture_client.api import RobotApi
from gesture_client.config import ClientSettings
from gesture_client.inference import YoloV5Detector
from gesture_client.model_cache import ModelCache
from gesture_client.ui import MainWindow
from gesture_server.app import Runtime, create_app
from gesture_server.arm import ArmController, DryRunArm
from gesture_server.camera import MockCamera, OpenCVCamera
from gesture_server.config import ServerSettings


def main():
    logging.getLogger('httpx').setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--mock-camera', action='store_true')
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--slow-inference', type=float, default=0)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.seconds < 8:
        parser.error('--seconds must be at least 8')
    os.chdir(root)
    qt = QApplication.instance() or QApplication([])
    camera = (MockCamera(640, 480, fps=15) if args.mock_camera else
              OpenCVCamera(args.camera, 640, 480, 15, 80))
    hardware = DryRunArm()
    runtime = Runtime(camera, ArmController(hardware))
    app = create_app(ServerSettings(dry_run=True, mock_camera=args.mock_camera), runtime)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    server_thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    server_thread.start()

    def pump_until(predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            qt.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        return False

    window = None
    try:
        if not pump_until(lambda: server.started or not server_thread.is_alive(), 10) or not server.started:
            raise RuntimeError('Local server failed to start; check camera availability')
        with tempfile.TemporaryDirectory(prefix='gesture-stream-test-') as directory:
            config = Path(directory) / 'settings.json'
            settings = ClientSettings(host='127.0.0.1', port=port)
            settings.save(config)
            window = MainWindow(config)
            window.show()
            class DelayedDetector(YoloV5Detector):
                def detect(self, bgr, render=True):
                    if args.slow_inference:
                        time.sleep(args.slow_inference)
                    result = super().detect(bgr, render=render)
                    result.elapsed_ms += args.slow_inference * 1000
                    return result
            window.model_cache = ModelCache(DelayedDetector)
            displayed = []
            show_frame = window._show_frame
            def record_frame(bgr):
                show_frame(bgr)  # actual BGR conversion, QImage, QPixmap and scaling
                displayed.append(time.monotonic())
            window._show_frame = record_frame
            logs = []
            started = time.monotonic()
            window.start_pipeline()
            worker = window.worker
            worker.log_message.connect(logs.append)
            pump_until(lambda: time.monotonic() - started >= args.seconds, args.seconds + 1)
            _, _, metrics = worker.snapshot()
            job, reused = window.model_cache.acquire(window.settings)
            assert job.ready.is_set() and job.error is None, job.error
            assert displayed and metrics['inferred_frames'] > 0, 'No video or inference'
            steady_start = started + job.elapsed_s + 1
            steady_frames = [t for t in displayed if t >= steady_start]
            steady_elapsed = max(0.001, time.monotonic() - steady_start)
            display_fps = len(steady_frames) / steady_elapsed
            pauses = [b - a for a, b in zip(steady_frames, steady_frames[1:])]
            first_preview_s = displayed[0] - started
            assert display_fps >= 7, f'Preview too slow: {display_fps:.1f} FPS'
            if args.slow_inference >= 0.3:
                assert display_fps > metrics['inference_fps'] * 2, 'Preview still blocked by inference'
            window.stop_pipeline()
            assert pump_until(lambda: window.worker is None, 10), 'Worker did not stop cleanly'

            displayed.clear()
            restarted_at = time.monotonic()
            window.start_pipeline()
            assert pump_until(lambda: bool(displayed), 5), 'Restart received no preview'
            cached_preview_s = displayed[0] - restarted_at
            assert pump_until(lambda: window.worker.snapshot()[2]['inferred_frames'] > 0, 5)
            window.stop_pipeline()
            assert pump_until(lambda: window.worker is None, 10)
            assert pump_until(lambda: not runtime.arm.status.busy, 3)
            api = RobotApi(settings.base_url)
            try:
                action = api.action('ok', 1, 0, 0)
                assert action.accepted
                assert pump_until(lambda: 'action:ok' in hardware.calls, 3)
            finally:
                api.close()
            report = {
                'camera': 'mock' if args.mock_camera else args.camera,
                'robot': 'mock', 'seconds': args.seconds,
                'artificial_inference_delay_ms': args.slow_inference * 1000,
                'first_preview_s': round(first_preview_s, 3),
                'cached_restart_preview_s': round(cached_preview_s, 3),
                'steady_display_fps': round(display_fps, 2),
                'max_steady_preview_gap_ms': round(max(pauses, default=0) * 1000, 1),
                'model_cache_reused': reused, 'mock_action_verified': 'action:ok' in hardware.calls,
                'metrics': metrics, 'logs': logs,
            }
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            window.close()
    finally:
        if window is not None and window.worker is not None:
            window.stop_pipeline()
            pump_until(lambda: window.worker is None, 10)
        server.should_exit = True
        server_thread.join(timeout=8)
        sock.close()
        if server_thread.is_alive():
            raise RuntimeError('Local server did not stop')


if __name__ == '__main__':
    main()
