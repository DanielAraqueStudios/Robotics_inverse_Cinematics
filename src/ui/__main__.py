"""Punto de entrada: python -m ui   (desde la carpeta src, con la API en ejecución)."""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    qss = Path(__file__).with_name("style.qss")
    app.setStyleSheet(qss.read_text(encoding="utf-8"))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
