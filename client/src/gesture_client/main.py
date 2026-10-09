import sys
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .ui import MainWindow


def run() -> None:
    logging.getLogger('httpx').setLevel(logging.WARNING)
    app = QApplication(sys.argv)
    window = MainWindow(Path.cwd() / "client-settings.json")
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    run()

