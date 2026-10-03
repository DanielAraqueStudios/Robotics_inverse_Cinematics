"""Ventana principal: modo FK (ángulos -> pose) e IK (pose objetivo -> ángulos)."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .api_client import ApiClient
from .presets import PRESETS
from .steps_tab import StepsTab
from .viewer import ChainViewer

DEFAULT_URL = "http://127.0.0.1:8000"
SLIDER_SCALE = 100
REASON_TEXT = {
    "ok": "Solución encontrada",
    "unreachable": "El objetivo está fuera del alcance del robot",
    "max_iter": "El cálculo no convergió dentro del límite de iteraciones",
    "out_of_limits": "La solución choca con los límites articulares",
    "singular": "El objetivo está en una configuración singular",
}


class MainWindow(QMainWindow):
    def __init__(self, base_url: str = DEFAULT_URL):
        super().__init__()
        self.setWindowTitle("Cinemática — FK / IK")
        self.resize(1280, 800)
        self.client = ApiClient(base_url, self)
        self.client.bridge.finished.connect(self._on_result)
        self.client.bridge.failed.connect(self._on_failure)

        self._robot = {}
        self._q: list[float] = []
        self._joint_sliders: list[tuple[QSlider, QDoubleSpinBox]] = []
        self._solutions: list[dict] = []
        self._last_ik: dict | None = None
        self._fk_inflight = False
        self._fk_pending = False
        self._fk_timer = QTimer(self)
        self._fk_timer.setSingleShot(True)
        self._fk_timer.setInterval(60)
        self._fk_timer.timeout.connect(self._request_fk)

        self._build_ui()
        self._load_preset(next(iter(PRESETS)))

    # ----- construcción de la interfaz -----

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(12)

        outer.addLayout(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_controls(), 0)
        body.addWidget(self._build_viewport(), 1)
        body.addWidget(self._build_results(), 0)
        outer.addLayout(body, 1)

        self.setCentralWidget(root)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel("Cinemática del robot")
        title.setObjectName("title")
        row.addWidget(title)
        row.addStretch(1)

        self.preset_box = QComboBox()
        self.preset_box.addItems(PRESETS.keys())
        self.preset_box.currentTextChanged.connect(self._load_preset)
        row.addWidget(QLabel("Robot"))
        row.addWidget(self.preset_box)

        self.url_edit = QLineEdit(self.client.base_url)
        self.url_edit.setFixedWidth(220)
        self.url_edit.editingFinished.connect(self._apply_url)
        row.addWidget(QLabel("Servidor"))
        row.addWidget(self.url_edit)

        self.status_pill = QLabel("Listo")
        self.status_pill.setObjectName("pill")
        row.addWidget(self.status_pill)
        return row

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setFixedWidth(360)
        layout = QVBoxLayout(panel)

        self.fk_btn = QPushButton("Cinemática directa (FK)")
        self.ik_btn = QPushButton("Cinemática inversa (IK)")
        for btn in (self.fk_btn, self.ik_btn):
            btn.setCheckable(True)
            btn.setObjectName("mode")
        group = QButtonGroup(self)
        group.addButton(self.fk_btn, 0)
        group.addButton(self.ik_btn, 1)
        self.fk_btn.setChecked(True)
        group.idClicked.connect(self._set_mode)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.fk_btn)
        mode_row.addWidget(self.ik_btn)
        layout.addLayout(mode_row)

        self.mode_hint = QLabel()
        self.mode_hint.setObjectName("hint")
        self.mode_hint.setWordWrap(True)
        layout.addWidget(self.mode_hint)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_fk_page())
        self.stack.addWidget(self._build_ik_page())
        layout.addWidget(self.stack, 1)
        return panel

    def _build_fk_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._fk_container = QWidget()
        self._fk_layout = QVBoxLayout(self._fk_container)
        self._fk_layout.setSpacing(10)
        self._fk_layout.addStretch(1)
        scroll.setWidget(self._fk_container)
        return scroll

    def _build_ik_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        grid = QGridLayout()
        self.target_spins: list[QDoubleSpinBox] = []
        labels = [("x [m]", 0.0), ("y [m]", 0.0), ("z [m]", 0.0),
                  ("roll [°]", 0.0), ("pitch [°]", 0.0), ("yaw [°]", 0.0)]
        for i, (text, default) in enumerate(labels):
            spin = QDoubleSpinBox()
            spin.setRange(-360.0 if i >= 3 else -10.0, 360.0 if i >= 3 else 10.0)
            spin.setDecimals(3)
            spin.setValue(default)
            spin.setSingleStep(0.05 if i < 3 else 1.0)
            self.target_spins.append(spin)
            grid.addWidget(QLabel(text), i, 0)
            grid.addWidget(spin, i, 1)
        layout.addLayout(grid)

        self.solve_btn = QPushButton("Calcular cinemática inversa")
        self.solve_btn.setObjectName("primary")
        self.solve_btn.clicked.connect(self._request_ik)
        layout.addWidget(self.solve_btn)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setTextVisible(False)
        self.busy.setVisible(False)
        layout.addWidget(self.busy)

        layout.addWidget(QLabel("Soluciones (codo arriba / abajo)"))
        self.solutions_list = QListWidget()
        self.solutions_list.currentRowChanged.connect(self._show_solution)
        layout.addWidget(self.solutions_list, 1)
        return page

    def _build_viewport(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("viewport")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        self.viewer = ChainViewer(panel)
        layout.addWidget(self.viewer)
        hint = QLabel("Arrastre con el ratón para rotar la vista")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        self.tabs = QTabWidget()
        self.tabs.addTab(panel, "Vista 3D")
        self.steps_tab = StepsTab()
        self.tabs.addTab(self.steps_tab, "Solución paso a paso")
        return self.tabs

    def _build_results(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        layout.addWidget(self._section("Estado"))
        self.status_label = QLabel("Mueva las articulaciones para ver la pose.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.warning_label = QLabel("")
        self.warning_label.setObjectName("warning")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)

        layout.addWidget(self._section("Efector final"))
        self.readout = QLabel("—")
        self.readout.setObjectName("readout")
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)

        self.steps_btn = QPushButton("Show step-by-step solution")
        self.steps_btn.setObjectName("primary")
        self.steps_btn.clicked.connect(self._request_steps)
        layout.addWidget(self.steps_btn)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setObjectName("section")
        return lbl

    # ----- robot y articulaciones -----

    def _load_preset(self, name: str) -> None:
        self._robot = PRESETS[name]
        self._q = []
        self._joint_sliders = []
        while self._fk_layout.count() > 1:
            item = self._fk_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, joint in enumerate(self._robot["joints"]):
            self._fk_layout.insertWidget(i, self._make_joint_row(i, joint))
            self._q.append(0.0)
        self._set_mode(0 if self.fk_btn.isChecked() else 1)
        self._request_fk()

    def _make_joint_row(self, index: int, joint: dict) -> QWidget:
        is_rev = joint["kind"] == "R"
        unit = "°" if is_rev else "m"
        if is_rev:
            lo, hi = -180.0, 180.0
        else:
            lo = joint.get("lower") if joint.get("lower") is not None else -1.0
            hi = joint.get("upper") if joint.get("upper") is not None else 1.0

        box = QWidget()
        box.setObjectName("joint")
        row = QVBoxLayout(box)
        head = QHBoxLayout()
        kind = "Rotacional" if is_rev else "Prismática"
        head.addWidget(QLabel(f"θ{index + 1}" if is_rev else f"d{index + 1}"))
        head.addWidget(QLabel(kind))
        head.addStretch(1)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(2 if is_rev else 3)
        spin.setSuffix(f" {unit}")
        spin.setFixedWidth(120)
        spin.setValue(0.0 if lo <= 0.0 <= hi else lo)
        head.addWidget(spin)
        row.addLayout(head)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(lo * SLIDER_SCALE), int(hi * SLIDER_SCALE))
        slider.setValue(int(spin.value() * SLIDER_SCALE))
        row.addWidget(slider)

        slider.valueChanged.connect(lambda v, s=spin: s.setValue(v / SLIDER_SCALE))
        spin.valueChanged.connect(lambda v, s=slider, i=index, r=is_rev: self._on_joint_changed(i, v, r, s))
        self._joint_sliders.append((slider, spin))
        return box

    def _on_joint_changed(self, index: int, value: float, is_rev: bool, slider: QSlider) -> None:
        if slider.value() != int(round(value * SLIDER_SCALE)):
            slider.blockSignals(True)
            slider.setValue(int(round(value * SLIDER_SCALE)))
            slider.blockSignals(False)
        self._q[index] = math.radians(value) if is_rev else value
        self._fk_timer.start()

    def _set_q(self, q: list[float]) -> None:
        """Escribe q en los controles sin disparar nuevas solicitudes."""
        for i, (slider, spin) in enumerate(self._joint_sliders):
            is_rev = self._robot["joints"][i]["kind"] == "R"
            value = math.degrees(q[i]) if is_rev else q[i]
            spin.blockSignals(True)
            slider.blockSignals(True)
            spin.setValue(value)
            slider.setValue(int(round(spin.value() * SLIDER_SCALE)))
            spin.blockSignals(False)
            slider.blockSignals(False)
            self._q[i] = q[i]

    # ----- modos -----

    def _set_mode(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        if idx == 0:
            self.mode_hint.setText("FK: fije los ángulos y la pose resultante se muestra en la vista.")
        else:
            self.mode_hint.setText("IK: fije la pose objetivo y se calcularán los ángulos articulares.")

    def _apply_url(self) -> None:
        self.client.base_url = self.url_edit.text().strip() or DEFAULT_URL

    # ----- solicitudes a la API -----

    def _robot_payload(self) -> dict:
        return self._robot

    def _request_fk(self) -> None:
        if self._fk_inflight:
            self._fk_pending = True
            return
        self._fk_inflight = True
        self.client.request("fk", "/api/fk", {"robot": self._robot_payload(), "q": list(self._q)})
        self._set_status("Calculando pose…", busy=True)

    def _request_ik(self) -> None:
        vals = [spin.value() for spin in self.target_spins]
        seed = list(self._q)
        self._last_ik = {"robot": self._robot_payload(), "position": vals[:3], "rpy_deg": vals[3:], "seed": seed}
        self.solve_btn.setEnabled(False)
        self.busy.setVisible(True)
        self._set_status("Resolviendo cinemática inversa…", busy=True)
        self.client.request("ik", "/api/ik", self._last_ik)

    def _request_steps(self) -> None:
        if self.stack.currentIndex() == 1:
            if self._last_ik is None:
                self._set_status("Solve an inverse-kinematics problem first.", error=True)
                return
            self.client.request("steps_ik", "/api/steps/ik", self._last_ik)
        else:
            self.client.request("steps_fk", "/api/steps/fk", {"robot": self._robot_payload(), "q": list(self._q)})
        self._set_status("Preparing step-by-step solution…", busy=True)

    def _on_result(self, tag: str, data: object) -> None:
        if tag == "fk":
            self._fk_inflight = False
            self._show_pose(data, label="Pose calculada")
            if self._fk_pending:
                self._fk_pending = False
                self._request_fk()
        elif tag == "ik":
            self._finish_ik(data)
        elif tag in ("steps_fk", "steps_ik"):
            title = "Forward kinematics" if tag == "steps_fk" else "Inverse kinematics"
            self.steps_tab.set_steps(data["steps"])
            self.tabs.setCurrentWidget(self.steps_tab)
            self._set_status(f"{title}: solución paso a paso lista", error=False)

    def _on_failure(self, tag: str, message: str) -> None:
        if tag == "fk":
            self._fk_inflight = False
        if tag == "ik":
            self.solve_btn.setEnabled(True)
            self.busy.setVisible(False)
        self._set_status(message, error=True)

    def _show_pose(self, chain: dict, label: str) -> None:
        self.viewer.draw_pose(chain["origins"])
        pos = chain["end_position"]
        rot = chain["rotation"]
        roll, pitch, yaw = _rotation_to_rpy_deg(rot)
        self.readout.setText(
            f"x = {pos[0]:+.4f} m\n"
            f"y = {pos[1]:+.4f} m\n"
            f"z = {pos[2]:+.4f} m\n\n"
            f"roll  = {roll:+.2f}°\n"
            f"pitch = {pitch:+.2f}°\n"
            f"yaw   = {yaw:+.2f}°"
        )
        singular = chain["singularity"] == "SINGULAR"
        self.viewer.set_warning(singular)
        self._set_warning(chain["singularity_message"] if singular else "")
        self._set_status(label, error=False)

    def _finish_ik(self, data: dict) -> None:
        self.solve_btn.setEnabled(True)
        self.busy.setVisible(False)
        self.solutions_list.clear()
        if not data["converged"]:
            text = REASON_TEXT.get(data["reason"], "No se encontró solución")
            self._set_status(text, error=True)
            self._set_warning("")
            return
        for k, sol in enumerate(data["solutions"], start=1):
            label = f"Solución {k}: " + ", ".join(
                f"{math.degrees(v):+.1f}°" if self._robot["joints"][i]["kind"] == "R" else f"{v:+.3f} m"
                for i, v in enumerate(sol["q"])
            )
            self.solutions_list.addItem(label)
        self._solutions = data["solutions"]
        self.solutions_list.setCurrentRow(0)
        self._show_solution(0)
        msg = f"{REASON_TEXT['ok']} ({len(data['solutions'])} configuración(es))."
        if data["singular_flag"]:
            msg += " Atención: configuración singular."
        self._set_status(msg, error=False)

    def _show_solution(self, row: int) -> None:
        if row < 0 or row >= len(self._solutions):
            return
        sol = self._solutions[row]
        self._set_q(sol["q"])
        self._show_pose(sol["chain"], label=f"Solución {row + 1} aplicada")

    # ----- retroalimentación -----

    def _set_status(self, text: str, busy: bool = False, error: bool = False) -> None:
        self.status_label.setText(text)
        self.status_pill.setText("Error" if error else ("Calculando" if busy else "Listo"))
        self.status_pill.setProperty("state", "error" if error else ("busy" if busy else "ok"))
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)

    def _set_warning(self, text: str) -> None:
        self.warning_label.setText(text)
        self.warning_label.setVisible(bool(text))


def _rotation_to_rpy_deg(R) -> tuple[float, float, float]:
    """Ángulos roll, pitch, yaw (ZYX) en grados a partir de una matriz de rotación."""
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2][0]))))
    roll = math.degrees(math.atan2(R[2][1], R[2][2]))
    yaw = math.degrees(math.atan2(R[1][0], R[0][0]))
    return roll, pitch, yaw
