from __future__ import annotations

import time
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_GESTURES = {"five", "ok", "six", "like", "yes"}
REFERENCE_WEIGHTS = Path(
    "example/Dofbot_YOLO_Qt_Integrated_20260924_v3_recognition_pose/"
    "weights/gesture_yolov5s_final.pt"
)


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
        # A fresh checkout already contains the reference weight under example/.
        # Use it when the documented models/ copy has not been created yet.
        if not weights.is_file() and REFERENCE_WEIGHTS.is_file():
            weights = REFERENCE_WEIGHTS.resolve()
        source = Path(source_path).resolve()
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
        raw_model = attempt_load(str(weights), map_location=device)
        self.model = AutoShape(raw_model)
        copy_attr(self.model, raw_model,
                  include=("yaml", "nc", "hyp", "names", "stride"))
        self.model.names = {0: "five", 1: "ok", 2: "six", 3: "like", 4: "yes"}
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
