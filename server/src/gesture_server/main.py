import argparse
import logging

import uvicorn

from .app import create_app
from .config import ServerSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture robot server")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--camera")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--jpeg-quality", type=int)
    parser.add_argument("--start-pose", choices=("recognition", "safe"))
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--mock-camera", action="store_true", default=None)
    return parser.parse_args()


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    overrides = {key: value for key, value in vars(args).items() if value is not None}
    settings = ServerSettings(**overrides)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()

