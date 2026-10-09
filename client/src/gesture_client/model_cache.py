"""Background model loading retained across start/stop cycles of one window."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .config import ClientSettings
from .inference import YoloV5Detector


class ModelJob:
    def __init__(self):
        self.ready = threading.Event()
        self.detector: YoloV5Detector | None = None
        self.error: Exception | None = None
        self.elapsed_s = 0.0


class ModelCache:
    def __init__(self, factory=YoloV5Detector):
        self.factory = factory
        self._lock = threading.Lock()
        self._job: ModelJob | None = None
        self._key = None
        # Also protects the temporary pathlib compatibility override on Windows.
        self._load_lock = threading.Lock()

    def acquire(self, settings: ClientSettings) -> tuple[ModelJob, bool]:
        path = Path(settings.weights_path).resolve()
        if not path.is_file():
            path = Path('models/best.pt').resolve()
        signature = (path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None
        key = (str(path), signature, str(Path(settings.yolo_source).resolve()),
               settings.confidence, settings.inference_size, settings.cpu_threads)
        with self._lock:
            if self._key == key and self._job is not None and self._job.error is None:
                return self._job, True
            job = ModelJob()
            self._job, self._key = job, key
            threading.Thread(target=self._load, args=(job, settings),
                             name='model-loader', daemon=True).start()
            return job, False

    def _load(self, job: ModelJob, settings: ClientSettings) -> None:
        started = time.monotonic()
        try:
            with self._load_lock:
                job.detector = self.factory(settings.weights_path, settings.yolo_source,
                    settings.confidence, settings.inference_size, settings.cpu_threads)
                if hasattr(job.detector, 'warmup'):
                    job.detector.warmup()
        except Exception as exc:
            job.error = exc
        finally:
            job.elapsed_s = time.monotonic() - started
            job.ready.set()
