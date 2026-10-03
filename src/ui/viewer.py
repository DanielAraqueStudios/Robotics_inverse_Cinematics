"""Visor 3D de la cadena cinemática (Matplotlib embebido) con animación entre poses."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QSizePolicy

BG = "#0F141B"
PANEL = "#151C25"
LINK = "#4CC9F0"
JOINT = "#E2E8F0"
TIP = "#F72585"
GRID = "#2A3441"
TEXT = "#94A3B8"
WARN = "#FFB703"

ANIM_FRAMES = 18
ANIM_INTERVAL_MS = 16


class ChainViewer(FigureCanvasQTAgg):
    """Dibuja los orígenes de cada eslabón que devuelve la API.

    Los datos mostrados son exactamente los de la respuesta del backend; la
    animación solo interpola entre la pose anterior y la nueva, sin inventar
    posiciones intermedias fuera de esa línea.
    """

    def __init__(self, parent=None):
        self.figure = Figure(facecolor=BG)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self._current = None  # np.ndarray (n+1, 3)
        self._start = None
        self._target = None
        self._frame = 0
        self._warning = False
        self._timer = QTimer(self)
        self._timer.setInterval(ANIM_INTERVAL_MS)
        self._timer.timeout.connect(self._step)
        self._style_axes()
        self.draw_pose(np.zeros((1, 3)), animate=False)

    def _style_axes(self) -> None:
        ax = self.ax
        ax.set_facecolor(PANEL)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            axis.label.set_color(TEXT)
            axis._axinfo["grid"]["color"] = GRID
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

    def set_warning(self, active: bool) -> None:
        self._warning = active
        self._redraw()

    def draw_pose(self, origins, animate: bool = True) -> None:
        """Muestra una pose nueva. origins: lista de puntos (base, articulaciones..., efector)."""
        target = np.asarray(origins, dtype=float).reshape(-1, 3)
        if self._current is None or not animate or self._current.shape != target.shape:
            self._timer.stop()
            self._current = target.copy()
            self._redraw()
            return
        self._start = self._current.copy()
        self._target = target
        self._frame = 0
        self._timer.start()

    def _step(self) -> None:
        self._frame += 1
        t = min(self._frame / ANIM_FRAMES, 1.0)
        t = t * t * (3 - 2 * t)  # suavizado smoothstep
        self._current = self._start + (self._target - self._start) * t
        self._redraw()
        if self._frame >= ANIM_FRAMES:
            self._timer.stop()

    def _redraw(self) -> None:
        ax = self.ax
        ax.cla()
        self._style_axes()
        pts = self._current
        if pts is None or len(pts) == 0:
            self.draw_idle()
            return

        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=LINK, linewidth=3.0, solid_capstyle="round")
        if len(pts) > 2:
            ax.scatter(pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2], color=JOINT, s=45, depthshade=False)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=TEXT, s=70, marker="s", depthshade=False)
        tip_color = WARN if self._warning else TIP
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=tip_color, s=110, depthshade=False)

        extent = max(float(np.max(np.abs(pts))), 0.5) * 1.2
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        ax.set_box_aspect((1, 1, 1))
        self.draw_idle()
