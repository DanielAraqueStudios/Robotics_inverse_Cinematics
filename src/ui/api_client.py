"""Cliente HTTP no bloqueante: las llamadas se ejecutan en un QThreadPool."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

REQUEST_TIMEOUT_S = 10.0


class ApiError(Exception):
    """Error de comunicación o de validación, con mensaje legible en español."""


def post_json(base_url: str, path: str, payload: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ApiError(_detail_from_http_error(exc)) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ApiError(
            "No se pudo conectar con el servidor de cinemática. "
            "Verifique que la API esté en ejecución."
        ) from exc


def _detail_from_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8"))
        detail = body.get("detail", "")
    except (ValueError, OSError):
        detail = ""
    if exc.code == 504:
        return "El cálculo tardó demasiado. Pruebe con otra posición objetivo."
    if exc.code == 422:
        return f"Datos no válidos: {detail}" if isinstance(detail, str) else "Datos no válidos."
    return f"Error del servidor ({exc.code})."


class _Bridge(QObject):
    """Señales que viajan del hilo de trabajo al hilo de la interfaz."""

    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)


class _Task(QRunnable):
    def __init__(self, bridge: _Bridge, tag: str, base_url: str, path: str, payload: dict):
        super().__init__()
        self._bridge = bridge
        self._tag = tag
        self._base_url = base_url
        self._path = path
        self._payload = payload

    def run(self) -> None:
        try:
            result = post_json(self._base_url, self._path, self._payload)
        except ApiError as exc:
            self._bridge.failed.emit(self._tag, str(exc))
        else:
            self._bridge.finished.emit(self._tag, result)


class ApiClient(QObject):
    """Envía solicitudes sin bloquear la interfaz. Cada solicitud lleva una etiqueta."""

    def __init__(self, base_url: str, parent: QObject | None = None):
        super().__init__(parent)
        self.base_url = base_url
        self._pool = QThreadPool.globalInstance()
        self.bridge = _Bridge(self)

    def request(self, tag: str, path: str, payload: dict) -> None:
        self._pool.start(_Task(self.bridge, tag, self.base_url, path, payload))
