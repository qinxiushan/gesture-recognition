from __future__ import annotations

import time
import sys
import os
import pathlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_GESTURES = {"fist", "five", "heartSingle", "ok", "thumbUp"}
REFERENCE_WEIGHTS = Path("models/best.pt")


@dataclass
class Detection:
    label: str | None
    confidence: float
    x: float
    y: float
    elapsed_ms: float
    rendered: np.ndarray


class YoloV5Detector:
    def __init__(self, weights_path: str, source_path: str, confidence: float):
        import torch
        weights = Path(weights_path).resolve()
        # Fall back to the bundled reference weight when the configured file
        # does not exist yet.
        if not weights.is_file() and REFERENCE_WEIGHTS.is_file():
            weights = REFERENCE_WEIGHTS.resolve()
        source = Path(source_path).resolve()
        if not source.is_dir():
            bundled_source = Path("third_party/yolov5-6.2").resolve()
            if bundled_source.is_dir():
                source = bundled_source
        if not weights.is_file():
            raise FileNotFoundError(f"模型权重不存在: {weights}")
        if not source.is_dir():
            raise FileNotFoundError(f"YOLOv5 源码目录不存在: {source}")
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        # This repository contains a customized YOLOv5 whose DetectionModel
        # does not expose model.autoshape(); load and wrap it explicitly.
        from models.experimental import attempt_load
        from models.common import AutoShape
        from utils.torch_utils import copy_attr

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.name == "nt":
            original_posix_path = pathlib.PosixPath
            try:
                pathlib.PosixPath = pathlib.PurePosixPath
                raw_model = attempt_load(str(weights), device=device)
            finally:
                pathlib.PosixPath = original_posix_path
        else:
            raw_model = attempt_load(str(weights), device=device)
        self.model = AutoShape(raw_model)
        copy_attr(self.model, raw_model,
                  include=("yaml", "nc", "hyp", "names", "stride"))
        self.model.names = {0: "fist", 1: "five", 2: "heartSingle", 3: "ok", 4: "thumbUp"}
        self.model.conf = confidence
        self.model.to(device)
        self.model.eval()

    def detect(self, bgr: np.ndarray) -> Detection:
        started = time.perf_counter()
        results = self.model(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size=640)
        boxes = results.xyxy[0]
        label, confidence, x, y = None, 0.0, 0.0, 0.0
        if boxes is not None and len(boxes):
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            box = boxes[int(areas.argmax().item())]
            x1, y1, x2, y2, score, class_id = [float(v.item()) for v in box[:6]]
            candidate = self.model.names[int(class_id)]
            if candidate in SUPPORTED_GESTURES:
                label, confidence = candidate, score
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
        rendered = results.render()[0]
        rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        return Detection(label, confidence, x, y,
                         (time.perf_counter() - started) * 1000, rendered)
