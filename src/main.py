"""Punto de entrada único: arranca la API en segundo plano y abre la interfaz.

Uso (desde la carpeta src):
    python main.py
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
from PyQt6.QtWidgets import QApplication

from api.app import app as api_app
from ui.main_window import MainWindow

HEALTH_TIMEOUT_S = 10.0


def free_port() -> int:
    """Puerto libre en localhost, elegido por el sistema operativo."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_api(port: int) -> uvicorn.Server:
    """Inicia la API en un hilo daemon y espera a que responda."""
    config = uvicorn.Config(api_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="api-server", daemon=True).start()

    health = f"http://127.0.0.1:{port}/api/health"
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=1.0) as resp:
                if resp.status == 200:
                    return server
        except OSError:
            time.sleep(0.1)
    server.should_exit = True
    raise RuntimeError("La API no respondió dentro del tiempo límite.")


def main() -> int:
    port = free_port()
    server = start_api(port)

    qt_app = QApplication(sys.argv)
    qss = Path(__file__).with_name("ui") / "style.qss"
    qt_app.setStyleSheet(qss.read_text(encoding="utf-8"))

    window = MainWindow(base_url=f"http://127.0.0.1:{port}")
    window.show()
    try:
        return qt_app.exec()
    finally:
        server.should_exit = True


if __name__ == "__main__":
    sys.exit(main())
