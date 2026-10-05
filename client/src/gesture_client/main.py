import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from .ui import MainWindow


def run() -> None:
    app = QApplication(sys.argv)
    window = MainWindow(Path.cwd() / "client-settings.json")
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    run()

