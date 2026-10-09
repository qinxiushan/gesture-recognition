#!/usr/bin/env bash
set -eu

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ "$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "3.6" ]; then
    echo "ERROR: this deployment profile requires python3 3.6.x" >&2
    exit 1
fi

# Ubuntu's python3-venv package mainly supplies ensurepip. Bypass it and
# unpack the already downloaded pure-Python wheels with the standard library.
if python3 -m venv --without-pip --system-site-packages .venv; then
    RUNTIME_PYTHON="$PROJECT_DIR/.venv/bin/python"
    PACKAGE_DIR="$($RUNTIME_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    MODE=venv
else
    echo "venv module unavailable; using a project-local package directory" >&2
    RUNTIME_PYTHON=python3
    PACKAGE_DIR="$PROJECT_DIR/.jetson_packages"
    MODE=local
fi

"$RUNTIME_PYTHON" "$PROJECT_DIR/scripts/bootstrap_wheels.py" "$PACKAGE_DIR"

if [ "$MODE" = venv ]; then
    PYTHONPATH="$PROJECT_DIR/shared/src:$PROJECT_DIR/server/src" \
        "$RUNTIME_PYTHON" -c "import cv2, numpy, fastapi, uvicorn, gesture_shared, gesture_server; print('Jetson environment OK')"
    touch "$PROJECT_DIR/.venv/.offline-ready"
else
    PYTHONPATH="$PACKAGE_DIR:$PROJECT_DIR/shared/src:$PROJECT_DIR/server/src" \
        "$RUNTIME_PYTHON" -c "import cv2, numpy, fastapi, uvicorn, gesture_shared, gesture_server; print('Jetson environment OK')"
    touch "$PACKAGE_DIR/.offline-ready"
fi

echo "Environment ready. Start with: ./scripts/run_jetson.sh"
