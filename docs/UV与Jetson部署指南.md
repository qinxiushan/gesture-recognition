# uv 环境管理与 Jetson 部署指南

本文适用于本仓库当前架构：PC 运行 `gesture-client` 和 YOLOv5 推理，Jetson 只运行
`gesture-server`，负责摄像头、MJPEG 视频流和 DOFBOT 控制。

## 1. `.pytest_cache` 是什么

执行 `pytest` 或 `uv run pytest` 后，pytest 的内置 cache 插件会在项目根目录创建
`.pytest_cache/`。它保存跨测试运行的辅助状态，例如：

- 上一次失败的测试，用于 `pytest --lf`（只重跑上次失败项）；
- 已收集到的测试节点，用于 `pytest --ff`、`--nf` 等排序；
- 测试或插件通过 `cache` fixture 写入的少量 JSON 数据。

它不是业务数据，也不是 Python 虚拟环境。删除后不会影响源码或依赖，下一次运行 pytest
时会重新生成。本仓库已经在 `.gitignore` 中忽略它。

常用命令：

```bash
# 查看缓存内容
uv run pytest --cache-show

# 本次测试开始前清空缓存
uv run pytest --cache-clear

# 手工删除（Linux/Jetson）
rm -rf .pytest_cache

# 手工删除（PowerShell）
Remove-Item -Recurse -Force .pytest_cache
```

## 2. uv 如何管理本项目环境

### 2.1 四个核心对象

| 对象 | 作用 |
|---|---|
| `pyproject.toml` | 声明 Python 版本、直接依赖、开发依赖和 workspace 成员 |
| `uv.lock` | 锁定解析后的完整依赖版本和不同平台可用包 |
| `.venv/` | 当前机器实际使用的隔离环境，不提交 Git |
| uv 缓存 | 保存下载的 Python、wheel 和源码包，可供多个项目复用 |

根目录声明了 `shared`、`server`、`client` 三个 workspace 成员。workspace 共用一个
`uv.lock` 和默认 `.venv`，不是三个自动隔离的虚拟环境。`--package` 用来选择本次同步
或运行哪个成员，而不是切换到另一个 `.venv`。

### 2.2 `uv sync` 与 `uv run`

```bash
# 创建/更新默认 .venv，并同步根项目默认依赖
uv sync

# PC 开发：同步全部 workspace 包
uv sync --all-packages

# Jetson：只同步服务端及其 shared 依赖，不装客户端和开发依赖
uv sync --package gesture-server --no-dev

# 校验 pyproject.toml 与 uv.lock 一致，禁止部署时偷偷改锁文件
uv sync --package gesture-server --no-dev --locked

# 在项目环境中运行命令；uv 会先检查锁文件和环境
uv run gesture-client
uv run gesture-server --dry-run --mock-camera
uv run pytest

# 明确选择 workspace 成员运行
uv run --package gesture-server gesture-server --dry-run --mock-camera

# 不检查/同步环境，适合已经固定好的离线运行环境
uv run --no-sync gesture-server
```

通常不需要执行 `activate`。`uv run ...` 会直接使用项目 `.venv`。若希望像传统 venv
一样操作，也可以手工激活：

```bash
# Linux / Jetson
source .venv/bin/activate
python -V
deactivate

# Windows PowerShell
.venv\Scripts\Activate.ps1
python -V
deactivate
```

### 2.3 Python 版本与环境位置

```bash
uv python list                 # 查看系统和 uv 管理的 Python
uv python install 3.10         # 安装 uv 管理的 Python 3.10
uv python pin 3.10             # 为当前项目写入 Python 选择
uv venv --python 3.10          # 显式重建 .venv
uv run python -V               # 查看实际运行版本
uv run which python            # Linux 查看实际解释器路径
uv tree                        # 查看依赖树
uv lock --check                # 检查锁文件是否需要更新
uv cache dir                   # 查看 uv 缓存位置
uv cache prune                 # 清理不再需要的缓存
```

若需要完全独立的环境，可指定项目环境路径：

```bash
export UV_PROJECT_ENVIRONMENT=/opt/gesture-recognition/.venv
uv sync --package gesture-server --no-dev
```

不要对系统 Python 执行 `sudo uv pip install ...`，也不要把共享的系统 Python 当作 uv
项目环境；`uv sync` 默认是精确同步，目标环境中未声明的包可能被移除。

## 3. Jetson 部署前兼容性判断

先在 Jetson 执行：

```bash
cat /etc/nv_tegra_release 2>/dev/null || true
cat /etc/os-release
uname -m
python3 --version
ls -l /dev/video*
```

检查结果：

- 架构应为 `aarch64`；
- 当前项目要求 Python 3.10 或更高；
- 推荐 JetPack 6 / Ubuntu 22.04，其系统 Python 通常为 3.10；
- JetPack 5 常见系统 Python 为 3.8，不满足当前项目约束；优先升级 JetPack，而不是强行
  混装系统 `Arm_Lib` 与另一个 Python ABI；
- 至少应能看到目标 `/dev/videoN` 摄像头设备。

当前服务端不使用 PyTorch。不要在 Jetson 执行 `uv sync --all-packages`，否则 uv 会尝试
安装仅供 PC 客户端使用的 PySide6、torch、torchvision 等大型依赖。

## 4. 推荐部署：JetPack 6 + Python 3.10

### 4.1 系统准备

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates \
  python3 python3-venv python3-dev \
  libgl1 libglib2.0-0 v4l-utils

python3 --version
```

检查摄像头：

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --all
```

若当前用户无摄像头、串口或 I²C 权限：

```bash
sudo usermod -aG video,dialout,i2c "$USER"
```

执行后注销并重新登录，让组权限生效。不要长期使用 root 运行服务。

### 4.2 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

如果 Jetson 无法访问 `astral.sh`，可在联网电脑从 uv 官方 GitHub Releases 下载 Linux
`aarch64` 压缩包，复制 `uv`、`uvx` 到 Jetson 的 `$HOME/.local/bin/` 并赋予执行权限。

### 4.3 获取代码

```bash
cd "$HOME"
git clone https://github.com/qinxiushan/gesture-recognition.git
cd gesture-recognition
git switch main
```

### 4.4 处理 `Arm_Lib`

先验证厂商库在系统 Python 中是否可见：

```bash
python3 -c "import Arm_Lib; print(Arm_Lib.__file__)"
```

若成功，建议让项目 venv 继承系统 site-packages，以复用厂商安装的 `Arm_Lib`：

```bash
rm -rf .venv
uv venv --python /usr/bin/python3 --system-site-packages
```

若失败，应先按 DOFBOT 厂商说明安装 `Arm_Lib` 并再次执行导入检查。若手头有厂商提供的
wheel 或 Python 包，也可直接装入项目环境：

```bash
uv pip install /path/to/Arm_Lib-*.whl
uv run python -c "import Arm_Lib; print(Arm_Lib.__file__)"
```

不要简单复制一个为其他 Python 版本编译的 `.so` 文件；其 ABI 不匹配时会出现
`cannot open shared object`、`undefined symbol` 等错误。

### 4.5 安装服务端依赖

```bash
uv sync --package gesture-server --no-dev --locked
uv run python -c "import cv2, fastapi, gesture_server; print('server imports OK')"
uv run python -c "import Arm_Lib; print('Arm_Lib OK')"
```

如果锁文件中的某个 wheel 在 aarch64 不可用，先保留错误日志，不要改系统 Python。可用：

```bash
uv sync --package gesture-server --no-dev --locked -v
```

确定具体包后再处理。常见情况是 OpenCV wheel 不匹配；可安装 Jetson 系统 OpenCV，随后
使用 `--system-site-packages` 重建环境，并在项目中针对该平台调整依赖，而不是同时混用
多个 OpenCV 二进制包。

### 4.6 分阶段启动

第一阶段，不连接摄像头和机械臂：

```bash
uv run gesture-server \
  --host 0.0.0.0 --port 8896 \
  --dry-run --mock-camera
```

在 Jetson 本机验证：

```bash
curl http://127.0.0.1:8896/api/v1/health
```

第二阶段，使用真实摄像头但不动机械臂：

```bash
uv run gesture-server \
  --host 0.0.0.0 --port 8896 \
  --camera /dev/video0 --width 640 --height 480 --fps 15 \
  --dry-run
```

浏览器访问：

```text
http://JETSON_IP:8896/api/v1/health
http://JETSON_IP:8896/api/v1/video/stream
```

第三阶段，确认机械臂周围无障碍物、供电稳定后，去掉 `--dry-run`：

```bash
uv run gesture-server \
  --host 0.0.0.0 --port 8896 \
  --camera /dev/video0 --width 640 --height 480 --fps 15 \
  --start-pose recognition
```

## 5. systemd 开机启动

先手工运行并确认真实硬件正常，再创建服务：

```bash
sudo tee /etc/systemd/system/gesture-server.service >/dev/null <<'EOF'
[Unit]
Description=Gesture Recognition Robot Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/gesture-recognition
Environment=PATH=/home/jetson/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
Environment=GESTURE_SERVER_CAMERA=/dev/video0
Environment=GESTURE_SERVER_WIDTH=640
Environment=GESTURE_SERVER_HEIGHT=480
Environment=GESTURE_SERVER_FPS=15
Environment=GESTURE_SERVER_JPEG_QUALITY=80
Environment=GESTURE_SERVER_START_POSE=recognition
ExecStart=/home/jetson/.local/bin/uv run --no-sync gesture-server --host 0.0.0.0 --port 8896
Restart=on-failure
RestartSec=3
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
EOF
```

将示例中的用户和路径替换为真实值，然后：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gesture-server
systemctl status gesture-server --no-pager
journalctl -u gesture-server -f
```

升级代码时：

```bash
cd /home/jetson/gesture-recognition
git pull --ff-only
uv sync --package gesture-server --no-dev --locked
sudo systemctl restart gesture-server
```

## 6. 常见问题处理

### `uv: command not found`

```bash
export PATH="$HOME/.local/bin:$PATH"
which uv
```

将 PATH 设置加入 `~/.bashrc`，systemd 则使用服务文件中的绝对路径。

### `No interpreter found for Python >=3.10`

JetPack 6 可优先使用 `/usr/bin/python3`。否则尝试：

```bash
uv python install 3.10
uv venv --python 3.10
```

但 uv 管理的 Python 看不到只安装在系统 Python 中的 `Arm_Lib`。此时应把厂商包装入该
venv；如果厂商模块只支持系统 Python 3.8，应升级 JetPack或为项目做专门的 Python 3.8
兼容分支，不建议跨 ABI 强行加载。

### `No module named Arm_Lib`

```bash
python3 -c "import Arm_Lib; print(Arm_Lib.__file__)"
uv run python -c "import sys; print(sys.executable); print('\n'.join(sys.path))"
```

若第一条成功、第二条失败，使用系统 Python 重新创建带 `--system-site-packages` 的
`.venv`，或将厂商包安装到 `.venv`。

### 摄像头打不开

```bash
ls -l /dev/video*
fuser /dev/video0
v4l2-ctl --device=/dev/video0 --all
groups
```

检查设备编号、是否被其他进程占用、用户是否属于 `video` 组。CSI 相机若只提供 GStreamer
管线而不映射为 V4L2 设备，当前 `cv2.VideoCapture` 实现需要扩展 GStreamer source。

### 客户端无法连接 Jetson

```bash
hostname -I
ss -ltnp | grep 8896
curl http://127.0.0.1:8896/api/v1/health
```

确保 PC 与 Jetson 在同一网络，客户端填写 Jetson 实际 IP；若启用了 UFW：

```bash
sudo ufw allow from <PC所在网段/CIDR> to any port 8896 proto tcp
```

不要把无认证的 8896 端口直接暴露到公网。

### 依赖或环境损坏

```bash
rm -rf .venv
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --package gesture-server --no-dev --locked
```

若磁盘不足：

```bash
df -h
uv cache prune
```

### 只有把推理迁移到 Jetson 时才安装 PyTorch

本项目当前无需在 Jetson 安装 PyTorch。未来若迁移模型，必须先确认 JetPack 版本，再从
NVIDIA 的 Jetson PyTorch 兼容矩阵选择对应的 `linux_aarch64` wheel。不要直接使用当前
客户端约束的通用 PyPI `torch==2.5.x` 来期待 Jetson CUDA 加速。NVIDIA 官方流程要求先
安装 JetPack 和系统依赖，再安装与 JetPack 对应的 NVIDIA wheel，并用
`torch.cuda.is_available()` 验证。

## 7. 最终验收清单

```bash
uv --version
uv run python -V
uv run python -c "import cv2, fastapi, Arm_Lib; print('imports OK')"
curl http://127.0.0.1:8896/api/v1/health
systemctl is-active gesture-server  # 使用 systemd 时
```

- `health` 返回 `status: ok`；
- 摄像头 `available: true`；
- PC 能打开 MJPEG 流；
- `dry-run` 下五类动作请求均被服务端接受；
- 清空机械臂周围空间后再验证真实动作与安全姿态；
- 重启 Jetson 后服务能自动恢复，日志中没有持续摄像头或 `Arm_Lib` 错误。

## 8. 官方参考

- [uv 安装](https://docs.astral.sh/uv/getting-started/installation/)
- [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/)
- [uv 锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)
- [pytest 跨运行缓存](https://docs.pytest.org/en/stable/how-to/cache.html)
- [NVIDIA：在 Jetson 安装 PyTorch](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [NVIDIA：Jetson PyTorch 兼容信息](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)
