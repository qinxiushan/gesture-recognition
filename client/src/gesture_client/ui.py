from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import time

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
    QWidget,
)

from .config import ClientSettings
from .workers import PipelineWorker
from .model_cache import ModelCache


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.settings = ClientSettings.load(config_path)
        self.worker: PipelineWorker | None = None
        self.model_cache = ModelCache()
        self._closing = False
        self._shown_sequence = 0
        self._displayed = 0
        self._display_sample_at = time.monotonic()
        self.setWindowTitle("手势识别机械臂控制端")
        self.resize(1180, 760)
        self._build()
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._refresh)

    def _build(self) -> None:
        root = QWidget(); self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        connection = QGroupBox("连接与模型")
        form = QGridLayout(connection)
        self.host = QLineEdit(self.settings.host)
        self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(self.settings.port)
        self.weights = QLineEdit(self.settings.weights_path)
        browse = QPushButton("选择权重"); browse.clicked.connect(self._browse)
        self.start = QPushButton("连接并开始"); self.start.clicked.connect(self.start_pipeline)
        self.stop = QPushButton("停止"); self.stop.setEnabled(False); self.stop.clicked.connect(self.stop_pipeline)
        self.connection = QLabel("未连接")
        form.addWidget(QLabel("Jetson 地址"), 0, 0); form.addWidget(self.host, 0, 1)
        form.addWidget(QLabel("端口"), 0, 2); form.addWidget(self.port, 0, 3)
        form.addWidget(QLabel("模型权重"), 1, 0); form.addWidget(self.weights, 1, 1, 1, 2)
        form.addWidget(browse, 1, 3); form.addWidget(self.start, 0, 4)
        form.addWidget(self.stop, 1, 4); form.addWidget(self.connection, 0, 5, 2, 1)
        layout.addWidget(connection)

        body = QHBoxLayout(); layout.addLayout(body, 1)
        self.video = QLabel("等待视频流…"); self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(720, 540); self.video.setStyleSheet("background:#111;color:#aaa")
        body.addWidget(self.video, 3)
        side = QVBoxLayout(); body.addLayout(side, 1)
        result = QGroupBox("识别结果"); result_form = QFormLayout(result)
        self.gesture = QLabel("none"); self.gesture.setStyleSheet("font-size:26px;font-weight:bold;color:#159447")
        self.confidence = QLabel("0.0%")
        self.elapsed = QLabel("0 ms")
        self.arm = QLabel("空闲")
        result_form.addRow("手势", self.gesture); result_form.addRow("置信度", self.confidence)
        result_form.addRow("推理耗时", self.elapsed); result_form.addRow("机械臂", self.arm)
        self.performance = QLabel("接收 / 显示 / 推理：等待数据")
        self.performance.setWordWrap(True)
        result_form.addRow("性能", self.performance)
        side.addWidget(result)
        tuning = QGroupBox("安全门控"); tune_form = QFormLayout(tuning)
        self.threshold = QDoubleSpinBox(); self.threshold.setRange(0.05, 0.99); self.threshold.setSingleStep(0.05); self.threshold.setValue(self.settings.confidence)
        self.stable = QSpinBox(); self.stable.setRange(1, 30); self.stable.setValue(self.settings.stable_frames)
        self.cooldown = QDoubleSpinBox(); self.cooldown.setRange(0, 60); self.cooldown.setValue(self.settings.cooldown_seconds)
        tune_form.addRow("置信度", self.threshold); tune_form.addRow("稳定帧数", self.stable); tune_form.addRow("冷却秒数", self.cooldown)
        self.inference_fps = QSpinBox(); self.inference_fps.setRange(1, 30)
        self.inference_fps.setValue(self.settings.inference_fps)
        self.inference_size = QSpinBox(); self.inference_size.setRange(320, 1280)
        self.inference_size.setSingleStep(32); self.inference_size.setValue(self.settings.inference_size)
        self.cpu_threads = QSpinBox(); self.cpu_threads.setRange(1, 32)
        self.cpu_threads.setValue(self.settings.cpu_threads)
        tune_form.addRow("推理上限 FPS", self.inference_fps)
        tune_form.addRow("推理尺寸", self.inference_size)
        tune_form.addRow("CPU 线程", self.cpu_threads)
        side.addWidget(tuning)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(500)
        side.addWidget(QLabel("运行日志")); side.addWidget(self.log, 1)

    def _browse(self) -> None:
        value, _ = QFileDialog.getOpenFileName(self, "选择 YOLOv5 权重", "", "PyTorch (*.pt)")
        if value: self.weights.setText(value)

    def _collect(self) -> ClientSettings:
        return replace(self.settings, host=self.host.text().strip(), port=self.port.value(),
                              weights_path=self.weights.text().strip(),
                              yolo_source=self.settings.yolo_source,
                              confidence=self.threshold.value(), stable_frames=self.stable.value(),
                              cooldown_seconds=self.cooldown.value(),
                              inference_fps=self.inference_fps.value(),
                              inference_size=self.inference_size.value(),
                              cpu_threads=self.cpu_threads.value())

    def start_pipeline(self) -> None:
        self.settings = self._collect(); self.settings.save(self.config_path)
        self.worker = PipelineWorker(self.settings, self.model_cache)
        self.worker.connection_changed.connect(self._show_connection)
        self.worker.log_message.connect(self.log.append)
        self.worker.finished.connect(self._finished)
        self.start.setEnabled(False); self.stop.setEnabled(True)
        self._shown_sequence = self._displayed = 0
        self._display_sample_at = time.monotonic()
        self.preview_timer.start(max(1, round(1000 / max(1, self.settings.display_fps))))
        self.worker.start()

    def _refresh(self) -> None:
        if self.worker is None:
            return
        frame, result, metrics = self.worker.snapshot(self._shown_sequence)
        now = time.monotonic()
        age = metrics['frame_age_ms']
        if age is None or age > self.settings.max_frame_age_seconds * 1000:
            self.video.setText('等待新的视频帧…')
        if frame is not None and now - frame.received_at <= self.settings.max_frame_age_seconds:
            bgr = frame.bgr
            if (result is not None and result['shape'] == bgr.shape and
                    now - result['observed_at'] <= self.settings.overlay_ttl_seconds):
                bgr = bgr.copy()
                for x1, y1, x2, y2, confidence, label in result['boxes']:
                    cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
                    cv2.putText(bgr, f'{label} {confidence:.0%}', (int(x1), max(15, int(y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
            self._show_frame(bgr)
            self._shown_sequence = frame.sequence
            self._displayed += 1
        if result is not None and now - result['observed_at'] <= self.settings.max_frame_age_seconds:
            self._show_detection(dict(result, arm_busy=metrics['arm_busy']))
        else:
            self.gesture.setText('none'); self.confidence.setText('0.0%')
            self.arm.setText('忙碌' if metrics['arm_busy'] else '空闲')
        elapsed = now - self._display_sample_at
        if elapsed >= 1.0:
            age_text = f'{age:.0f} ms' if age is not None else '无帧'
            self.performance.setText(
                f"接收 {metrics['receive_fps']:.1f} / 显示 {self._displayed / elapsed:.1f} / "
                f"推理 {metrics['inference_fps']:.1f} FPS\n"
                f"传输 {metrics['mbps']:.2f} Mbps · 解码 {metrics['decode_ms']:.1f} ms\n"
                f"接收后帧龄 {age_text} · 推理 {metrics['inference_ms']:.1f} ms\n"
                f"连接 {metrics['connection_ms']:.0f} ms · 模型 {metrics['model_load_s']:.2f} s")
            self._displayed = 0
            self._display_sample_at = now

    def stop_pipeline(self) -> None:
        if self.worker: self.worker.stop()
        self.stop.setEnabled(False)

    def _show_frame(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self.video.setPixmap(QPixmap.fromImage(image).scaled(self.video.size(),
                             Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def _show_detection(self, result: dict) -> None:
        self.gesture.setText(result["label"])
        self.confidence.setText(f'{result["confidence"]:.1%}')
        self.elapsed.setText(f'{result["elapsed_ms"]:.1f} ms')
        self.arm.setText("忙碌" if result["arm_busy"] else "空闲")

    def _show_connection(self, state: str) -> None:
        self.connection.setText(state)

    def _finished(self) -> None:
        self.preview_timer.stop()
        self.start.setEnabled(True); self.stop.setEnabled(False); self.worker = None
        if self._closing:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker:
            self._closing = True
            self.worker.stop()
            event.ignore()
            return
        event.accept()

