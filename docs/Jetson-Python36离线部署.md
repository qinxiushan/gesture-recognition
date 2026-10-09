# Jetson Python 3.6.9 离线部署

此方案只部署 Jetson 服务端。PC 客户端、模型推理、测试工具仍在开发机运行，不复制到
Jetson 的 Python 环境。Jetson 使用原生 `venv`，完全不需要 `uv`。

## 1. 在联网电脑准备离线包

仓库的 `wheelhouse/` 已预下载服务端需要的纯 Python wheel。若需重新生成，可在 Windows 或
Linux 联网电脑执行：

```bash
python -m pip download --only-binary=:all: \
  --platform manylinux2014_aarch64 --implementation cp \
  --python-version 3.6.9 --abi cp36m \
  -r requirements-jetson-py36.txt -d wheelhouse
```

若下载机的 pip 太旧、不识别这些选项，请先在联网电脑升级 pip。将整个项目（包含新生成的
`wheelhouse/`）复制到 Jetson。不要为 Jetson 下载
`opencv-python`、`numpy` 或 `Arm_Lib`；优先使用 JetPack/厂商已经安装的版本。
离线包还包含 Python 3.6 专用的 `dataclasses`、`contextvars` 回移包以及 `immutables`
aarch64 wheel，并包含 FastAPI 需要的 `contextlib2`；安装脚本会自动解包。
其中 `contextvars` 在 PyPI 仅提供源码包，仓库内的纯 Python wheel 已在联网电脑上预先构建；
重新制作 `wheelhouse/` 时需要先执行 `python -m pip wheel --no-deps contextvars==2.4 -w wheelhouse`。

## 2. 检查系统组件

```bash
python3 --version
python3 -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
python3 -c "import Arm_Lib; print(Arm_Lib.__file__)"
```

不需要安装 `python3-venv` 或 `pip`。安装脚本会先尝试 `venv --without-pip`；若系统连
`venv` 模块也没有，则把依赖解包到项目内 `.jetson_packages/`。若缺少 `cv2`，优先安装
JetPack 对应的 `python3-opencv` 包。只有真实机械臂
模式需要 `Arm_Lib`；`--dry-run` 联调可以不安装它。

## 3. 创建环境并启动

```bash
chmod +x scripts/setup_jetson_py36.sh scripts/run_jetson.sh
./scripts/setup_jetson_py36.sh

# 先做无硬件联调
./scripts/run_jetson.sh --dry-run --mock-camera

# 再连接真实摄像头和机械臂
./scripts/run_jetson.sh --camera 0 --width 640 --height 480 --fps 15
```

若可创建 venv，环境使用 `--system-site-packages`，以便复用 JetPack 的 `cv2`、`numpy` 和
厂商 `Arm_Lib`；否则使用项目内依赖目录。纯 Python Web 依赖均从 `wheelhouse/` 解包，
整个过程无需 pip、ensurepip 和网络。

## 4. systemd 启动命令

服务文件中的启动项可写为：

```ini
WorkingDirectory=/opt/gesture-recognition
ExecStart=/opt/gesture-recognition/scripts/run_jetson.sh --camera 0
```

Python 3.6 已停止安全维护，这套固定版本仅作为旧 JetPack 的兼容方案。长期建议升级到
带 Python 3.8+ 的 JetPack；升级后仍可继续使用 `venv`，再逐步恢复新版依赖。
