from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
    QWidget,
)

from .config import ClientSettings
from .workers import PipelineWorker


class MainWindow(QMainWindow):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.settings = ClientSettings.load(config_path)
        self.worker: PipelineWorker | None = None
        self.setWindowTitle("手势识别机械臂控制端")
        self.resize(1180, 760)
        self._build()

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
        side.addWidget(result)
        tuning = QGroupBox("安全门控"); tune_form = QFormLayout(tuning)
        self.threshold = QDoubleSpinBox(); self.threshold.setRange(0.05, 0.99); self.threshold.setSingleStep(0.05); self.threshold.setValue(self.settings.confidence)
        self.stable = QSpinBox(); self.stable.setRange(1, 30); self.stable.setValue(self.settings.stable_frames)
        self.cooldown = QDoubleSpinBox(); self.cooldown.setRange(0, 60); self.cooldown.setValue(self.settings.cooldown_seconds)
        tune_form.addRow("置信度", self.threshold); tune_form.addRow("稳定帧数", self.stable); tune_form.addRow("冷却秒数", self.cooldown)
        side.addWidget(tuning)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        side.addWidget(QLabel("运行日志")); side.addWidget(self.log, 1)

    def _browse(self) -> None:
        value, _ = QFileDialog.getOpenFileName(self, "选择 YOLOv5 权重", "", "PyTorch (*.pt)")
        if value: self.weights.setText(value)

    def _collect(self) -> ClientSettings:
        return ClientSettings(host=self.host.text().strip(), port=self.port.value(),
                              weights_path=self.weights.text().strip(),
                              yolo_source=self.settings.yolo_source,
                              confidence=self.threshold.value(), stable_frames=self.stable.value(),
                              cooldown_seconds=self.cooldown.value())

    def start_pipeline(self) -> None:
        self.settings = self._collect(); self.settings.save(self.config_path)
        self.worker = PipelineWorker(self.settings)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.detection_ready.connect(self._show_detection)
        self.worker.connection_changed.connect(self._show_connection)
        self.worker.log_message.connect(self.log.append)
        self.worker.finished.connect(self._finished)
        self.start.setEnabled(False); self.stop.setEnabled(True)
        self.worker.start()

    def stop_pipeline(self) -> None:
        if self.worker: self.worker.stop()
        self.stop.setEnabled(False)

    def _show_frame(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self.video.setPixmap(QPixmap.fromImage(image).scaled(self.video.size(),
                             Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _show_detection(self, result: dict) -> None:
        self.gesture.setText(result["label"])
        self.confidence.setText(f'{result["confidence"]:.1%}')
        self.elapsed.setText(f'{result["elapsed_ms"]:.1f} ms')
        self.arm.setText("忙碌" if result["arm_busy"] else "空闲")

    def _show_connection(self, state: str) -> None:
        self.connection.setText(state)

    def _finished(self) -> None:
        self.start.setEnabled(True); self.stop.setEnabled(False); self.worker = None

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker:
            self.worker.stop(); self.worker.wait(4000)
        event.accept()

