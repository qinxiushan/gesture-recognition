# 手势识别机械臂系统

Jetson 采集摄像头、通过 HTTP/TCP 提供 MJPEG 视频，并执行机械臂动作；PC 上的
PySide6 客户端显示视频、运行 YOLOv5 手势识别，再通过 HTTP 请求触发动作。

| 运行位置 | 环境 | 部署内容 |
|---|---|---|
| PC 客户端 | Python 3.10+、uv | Qt 界面、YOLOv5 源码、模型权重 |
| Jetson 服务端 | Python 3.6.9、离线脚本 | 摄像头、Web 服务、机械臂控制 |

本文优先介绍当前 Jetson 的 Python 3.6.9 离线部署，不要求设备安装 uv、pip 或
`python3-venv`。完整说明见 [Python 3.6 离线部署](docs/Jetson-Python36离线部署.md)；
现代 Python 环境及 uv 的使用方式见 [uv 与 Jetson 部署指南](docs/UV与Jetson部署指南.md)。

## 工程结构

- `client/`：PySide6 界面、视频接收、YOLOv5 推理与安全门控。
- `server/`：FastAPI、摄像头采集和机械臂控制。
- `shared/`：Pydantic API 契约与固定枚举。
- `models/`：本地模型权重目录（权重被 Git 忽略）。
- `third_party/yolov5-6.2/`：PC 客户端加载 `.pt` 权重所需的 YOLOv5 源码。
- `scripts/`：Jetson 离线安装、启动及 PC 本机验证脚本。
- `wheelhouse/`：Python 3.6 Jetson 服务端的离线依赖包。
- `tests/`：协议、门控、服务端和模拟硬件测试。

## PC 安装与本机联调

要求 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)。先关闭旧服务、旧客户端，
再在仓库根目录安装依赖；Windows 运行中的进程会锁住相关文件。

```powershell
uv sync --package gesture-client --extra local-server
uv run --no-sync python scripts/local_server.py --mock-camera
```

另开终端启动客户端：

```powershell
uv run --no-sync gesture-client
```

模拟服务地址使用 `127.0.0.1:8896`。客户端需要同时具备
`models/best.pt` 权重和 `third_party/yolov5-6.2/` 源码；
仅有 `.pt` 文件无法反序列化和运行 YOLOv5 模型。也可直接在界面中选择其他权重。
客户端配置保存在仓库根目录的 `client-settings.json`。

没有 Jetson 时，可用本机摄像头替代模拟视频，机械臂仍由脚本强制模拟：

```powershell
uv run --no-sync python scripts/local_server.py --camera 0 --fps 15
```

客户端填写 `127.0.0.1`、端口 `8896`。`local_server.py` 只用于 PC 测试，
默认监听本机地址、强制模拟机械臂，不需要安装 `Arm_Lib`。
依赖安装完成后使用 `--no-sync` 启动，避免两个终端重复同步正在使用的环境。
性能方案与验证步骤见 [本机摄像头与卡顿验证](docs/本机摄像头与卡顿验证.md)。

## Jetson 部署

### 1. 准备文件和系统组件

将项目复制到 Jetson，例如 `~/09_code`。服务端至少需要 `server/`、`shared/`、
`scripts/`、`wheelhouse/` 和 `requirements-jetson-py36.txt`。
保留设备自己的环境，不要从 Windows 复制 `.venv`、`.jetson_packages` 或 `client-settings.json`。
YOLOv5、模型权重和 Qt 客户端在 PC 上运行，不需要安装到 Jetson。

```bash
cd ~/09_code
python3 --version
uname -m
python3 -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

本离线方案适用于 Python 3.6.x、aarch64 的 Jetson，优先复用 JetPack 提供的 OpenCV、NumPy。
只有运行真实机械臂时才需要厂商库：

```bash
python3 -c "import Arm_Lib; print(Arm_Lib.__file__)"
```

如果缺少系统 OpenCV、NumPy 或 Arm_Lib，需要准备与设备 JetPack 匹配的系统或厂商包；
`wheelhouse/` 不包含它们。离线依赖包包含 `dataclasses` 等 Python 3.6 回移包，
首次部署不需要在设备上联网下载。

### 2. 首次创建运行环境

```bash
chmod +x scripts/setup_jetson_py36.sh scripts/run_jetson.sh
./scripts/setup_jetson_py36.sh
```

出现 `Jetson environment OK` 表示依赖导入检查通过。脚本优先创建
`venv --without-pip --system-site-packages`，以便复用系统摄像头和机械臂库。
如果设备没有 venv 模块，则自动将依赖解包到项目的 `.jetson_packages/`，仍可运行。
这两种方式均不需要 uv、pip、ensurepip 或额外安装 `python3-venv`。

### 3. 分阶段启动

每次只运行一个服务，切换模式前按 `Ctrl+C` 停止上一条命令。

```bash
# 模拟视频 + 模拟机械臂：先验证服务和网络
./scripts/run_jetson.sh --dry-run --mock-camera

# 真实摄像头 + 模拟机械臂：验证采集、传输和 PC 识别
./scripts/run_jetson.sh --dry-run --camera 0 --width 640 --height 480 --fps 15

# 真实摄像头 + 真实机械臂：完成前两步后再运行
./scripts/run_jetson.sh --camera 0 --width 640 --height 480 --fps 15
```

USB 摄像头 `--camera 0` 通常对应 `/dev/video0`，也可传入 `--camera /dev/video0`。
CSI 摄像头可能需要与设备匹配的 GStreamer 管线，不能直接假定其编号为 0。
真实机械臂启动后会进入识别姿态，关闭服务时尝试回到安全姿态。

### 4. 连接 PC 客户端

Jetson 与 PC 接入互通的局域网，Jetson 本身不需要连接外网。另开 Jetson 终端查看地址：

```bash
hostname -I
```

在 PC 客户端填写能从 PC 访问的 Jetson 地址、端口 `8896`，点击“连接并开始”。
连接真实 Jetson 时不要填 `127.0.0.1`，该地址代表 PC 本机。
可先在 PC 浏览器打开 `http://<Jetson地址>:8896/api/v1/health`，
检查 `camera.available` 和 `arm.available`；模拟机械臂模式也会报告可用。

服务默认监听 `0.0.0.0:8896`。需要调整视频质量时，启动命令可增加
`--jpeg-quality 60`；`--fps` 控制 JPEG 输出上限，实际帧率受摄像头和设备性能影响。

### 5. 更新已有部署

本轮性能改进没有增加 Jetson 依赖，已安装的离线环境可以复用：

1. 停止旧服务，将更新后的 `server/`、`shared/` 和 `scripts/` 覆盖到原项目目录。
2. 同时更新 PC 客户端；只更新客户端不能修复旧服务的重复发帧问题。
3. 重新启动 `./scripts/run_jetson.sh --dry-run --camera 0 --fps 15` 完成验证，再切换真实机械臂。

只有离线依赖清单或设备环境发生变化时，才需要更新 `wheelhouse/` 并重新运行安装脚本。
目前已完成本机摄像头验证和 Python 3.6 语法检查，真实 Jetson 的驱动、Wi-Fi 和机械臂需现场复测。

如果设备已使用 Python 3.10+ 并能安装 uv，可选择
[现代 Python 部署流程](docs/UV与Jetson部署指南.md)；不要在 Python 3.6 环境执行工作区的 `uv sync`。

## API

- `GET /api/v1/health`：服务、摄像头和机械臂健康状态。
- `GET /api/v1/status`：机械臂当前状态。
- `GET /api/v1/video/stream`：MJPEG 视频流。
- `POST /api/v1/actions`：提交识别动作。
- `POST /api/v1/arm/pose`：切换 `recognition`/`safe` 姿态。
- `GET /docs`：FastAPI 交互式接口文档。

## 测试与排障

```powershell
uv sync --package gesture-client --extra local-server --group dev
uv run --no-sync pytest
```

- 依赖同步前先关闭所有使用该环境的程序；旧环境同时安装过两种 OpenCV 包时，
  参考 [本机验证指南](docs/本机摄像头与卡顿验证.md) 中的重装步骤。
- 无视频：检查 `/api/v1/health`、摄像头占用，以及 Jetson 上的 `ls -l /dev/video*`。
  有 `v4l2-ctl` 时可执行 `v4l2-ctl --list-devices`，设备消失时检查 USB 连接和供电。
- 无机械臂动作：先运行 `--dry-run`，再确认 `Arm_Lib` 与 I²C/串口权限。
- 客户端模型加载失败：检查权重文件和 `yolo_source` 指向匹配的 YOLOv5 源码。
- 视频卡顿：分别查看接收、显示和推理 FPS。推理上限只控制识别频率，不限制视频显示；
  接收慢时检查摄像头与网络，推理慢时再调整推理尺寸、CPU 线程和推理上限。
