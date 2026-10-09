#!/usr/bin/env bash
set -eu

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
if [ -f "$PROJECT_DIR/.venv/.offline-ready" ]; then
    RUNTIME_PYTHON="$PROJECT_DIR/.venv/bin/python"
    export PYTHONPATH="$PROJECT_DIR/shared/src:$PROJECT_DIR/server/src${PYTHONPATH:+:$PYTHONPATH}"
elif [ -f "$PROJECT_DIR/.jetson_packages/.offline-ready" ]; then
    RUNTIME_PYTHON=python3
    export PYTHONPATH="$PROJECT_DIR/.jetson_packages:$PROJECT_DIR/shared/src:$PROJECT_DIR/server/src${PYTHONPATH:+:$PYTHONPATH}"
else
    echo "Environment not set up; run ./scripts/setup_jetson_py36.sh first" >&2
    exit 1
fi
exec "$RUNTIME_PYTHON" -m gesture_server.main "$@"
