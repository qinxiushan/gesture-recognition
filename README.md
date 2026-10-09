# 手势识别机械臂系统

环境管理、`.pytest_cache` 说明与 Jetson 完整部署/排障请阅读
[`docs/UV与Jetson部署指南.md`](docs/UV与Jetson部署指南.md)。

这是一个 PC + Jetson 的 Python 单仓库项目。Jetson 采集摄像头并提供 MJPEG 视频流，
PC 上的 PySide6 客户端运行 YOLOv5 手势识别，再通过 REST 请求触发 DOFBOT 动作。

## 工程结构

- `client/`：PySide6 界面、视频接收、YOLOv5 推理与安全门控。
- `server/`：FastAPI、摄像头采集和机械臂控制。
- `shared/`：Pydantic API 契约与固定枚举。
- `models/`：本地模型权重目录（权重被 Git 忽略）。
- `tests/`：协议、门控、服务端和模拟硬件测试。

## 安装与模拟联调

要求 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)。在仓库根目录执行：

```powershell
uv sync --all-packages
uv run gesture-server --dry-run --mock-camera
```

另开终端启动客户端：

```powershell
uv run gesture-client
```

模拟服务地址使用 `127.0.0.1:8896`。首次运行客户端前，将参考权重复制到默认目录：

```powershell
Copy-Item example\Dofbot_YOLO_Qt_Integrated_20260924_v3_recognition_pose\weights\gesture_yolov5s_final.pt models\gesture_yolov5s_final.pt
```

也可直接在界面中选择权重。客户端配置保存在仓库根目录的 `client-settings.json`。

## Jetson 部署

将仓库复制到 Jetson，确认摄像头设备和厂商 `Arm_Lib` 已可被系统 Python 导入，然后执行：

```bash
uv sync --package gesture-server
uv run gesture-server --camera 0 --width 640 --height 480 --fps 15
```

可使用环境变量替代命令行参数，例如：

```bash
export GESTURE_SERVER_CAMERA=/dev/video0
export GESTURE_SERVER_JPEG_QUALITY=80
export GESTURE_SERVER_START_POSE=recognition
uv run gesture-server
```

真实硬件启动时默认使用识别姿态，关闭服务时尝试回到安全姿态。先用
`--dry-run --mock-camera` 完成网络联调，再去掉参数连接真实硬件。

## API

- `GET /api/v1/health`：服务、摄像头和机械臂健康状态。
- `GET /api/v1/status`：机械臂当前状态。
- `GET /api/v1/video/stream`：MJPEG 视频流。
- `POST /api/v1/actions`：提交识别动作。
- `POST /api/v1/arm/pose`：切换 `recognition`/`safe` 姿态。
- `GET /docs`：FastAPI 交互式接口文档。

## 测试与排障

```powershell
uv run pytest
```

- 无视频：检查 `/api/v1/health` 中的摄像头状态、设备路径和占用情况。
- 无机械臂动作：先运行 `--dry-run`，再确认 `Arm_Lib` 与 I²C/串口权限。
- 客户端模型加载失败：检查权重文件和 `yolo_source` 指向匹配的 YOLOv5 源码。
- 延迟增加：客户端只处理最新帧；可降低服务端分辨率、帧率或 JPEG 质量。
