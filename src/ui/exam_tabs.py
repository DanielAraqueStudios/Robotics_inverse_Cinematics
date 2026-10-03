"""Pestañas del análisis de examen (respuesta de POST /api/exam/analyze)."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

BG = "#0F141B"
PANEL = "#151C25"
ACCENT = "#4CC9F0"
ACCENT_BG = "#0C2A3A"
TIP = "#F72585"
OK = "#34D399"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
GRID = "#2A3441"
AXIS_COLORS = (("x_axis", "#EF4444"), ("y_axis", "#22C55E"), ("z_axis", "#3B82F6"))

TOL = 1e-6  # tolerancia de ortogonalidad y det R
TOL_IK = 1e-4  # tolerancia del error IK -> FK [m]

REQUIRED_KEYS = (
    "clasificacion", "grados_libertad", "marcos", "tabla_dh", "matrices",
    "fk", "jacobiano", "singularidad", "verificacion",
)
DH_HEADERS = ["i", "α(i-1)", "a(i-1)", "d(i)", "θ(i)", "tipo", "variable"]
JAC_ROWS = ["vx", "vy", "vz", "ωx", "ωy", "ωz"]
DH_NOTE = (
    "Convención de Craig (DH modificado): T(i-1,i) = Rot_x(α(i-1))·Trans_x(a(i-1))"
    "·Rot_z(θ(i))·Trans_z(d(i)). En cian, la variable articular de cada fila; "
    "atenuadas, las constantes."
)
JAC_NOTE = (
    "Articulaciones rotacionales (R): Jv = z × (o_n − o_i) y Jω = z. "
    "Prismáticas (P): Jv = z y Jω = 0. o_n es el origen del efector final."
)


# ----- utilidades de construcción -----

def _num(value: float) -> str:
    return f"{float(value):.4f}"


def _vec(values) -> str:
    return "[" + ", ".join(f"{float(v):+.3f}" for v in values) + "]"


def _rows(matrix) -> list[list[str]]:
    return [[_num(v) for v in row] for row in matrix]


def _list(values: list[int], prefix: str) -> str:
    return ", ".join(f"{prefix}{v}" for v in values) or "ninguna"


def _label(text: str, name: str | None = None, wrap: bool = True) -> QLabel:
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    lbl.setWordWrap(wrap)
    return lbl


def _badge(text: str, state: str) -> QLabel:
    lbl = _label(text, "badge", wrap=False)
    lbl.setProperty("state", state)
    return lbl


def _card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    if title:
        layout.addWidget(_label(title.upper(), "section", wrap=False))
    return frame, layout


def _matrix_card(title: str, matrix) -> QFrame:
    card, layout = _card()
    layout.addWidget(_label(title, "cardTitle", wrap=False))
    layout.addWidget(_matrix_table(_rows(matrix)))
    return card


def _page(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(content)
    return scroll


def _cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


def _readonly(table: QTableWidget) -> QTableWidget:
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
    table.resizeColumnsToContents()
    table.resizeRowsToContents()
    return table


def _matrix_table(rows: list[list[str]]) -> QTableWidget:
    table = QTableWidget(len(rows), len(rows[0]))
    table.setObjectName("matrix")
    table.horizontalHeader().setVisible(False)
    table.verticalHeader().setVisible(False)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            table.setItem(r, c, _cell(text))
    return _readonly(table)


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)


def _style_3d(ax) -> None:
    _style_axes(ax)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.set_xlabel("x [m]", color=MUTED)
    ax.set_ylabel("y [m]", color=MUTED)
    ax.set_zlabel("z [m]", color=MUTED)


def _frames_canvas(marcos: list[dict]) -> FigureCanvasQTAgg:
    """Dibuja los marcos: ejes X (rojo), Y (verde) y Z (azul) desde cada origen."""
    fig = Figure(facecolor=BG)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setMinimumHeight(360)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax)
    origins = np.array([m["origin"] for m in marcos], dtype=float).reshape(-1, 3)
    extent = max(float(np.max(np.abs(origins))) if origins.size else 0.0, 0.3) * 1.3
    length = extent * 0.25
    if origins.size:
        ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], color=MUTED, linewidth=1.5)
        ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], color=TEXT, s=30, depthshade=False)
    for m in marcos:
        o = np.asarray(m["origin"], dtype=float)
        for key, color in AXIS_COLORS:
            v = np.asarray(m[key], dtype=float) * length
            ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color=color, linewidth=2.0, arrow_length_ratio=0.2)
        ax.text(o[0], o[1], o[2], f" {m['name']}", color=TEXT, fontsize=8)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_box_aspect((1, 1, 1))
    return canvas


def _sv_canvas(values: list[float]) -> FigureCanvasQTAgg:
    fig = Figure(facecolor=BG)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setMinimumHeight(220)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    idx = np.arange(1, len(values) + 1)
    if len(values):
        ax.bar(idx, values, color=ACCENT, width=0.6)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"σ{i}" for i in idx])
    ax.set_title("Valores singulares", color=TEXT, fontsize=10)
    fig.tight_layout()
    return canvas


# ----- pestañas -----

def _robot_tab(data: dict) -> QWidget:
    clas = data["clasificacion"]
    body = QWidget()
    layout = QVBoxLayout(body)

    head = QHBoxLayout()
    head.addWidget(_badge(f"Configuración {clas['configuracion']}", "ok" if clas["familia"] == "RPRP" else "info"))
    head.addWidget(_badge(f"{data['grados_libertad']} GDL", "info"))
    head.addStretch(1)
    layout.addLayout(head)
    layout.addWidget(_label(clas["descripcion"], "hint"))

    card, card_layout = _card("Articulaciones")
    card_layout.addWidget(_label(f"Rotacionales: {_list(clas['rotacionales'], 'θ')}", "cardText"))
    card_layout.addWidget(_label(f"Prismáticas: {_list(clas['prismaticas'], 'd')}", "cardText"))
    layout.addWidget(card)

    card, card_layout = _card("Marcos de referencia")
    card_layout.addWidget(_frames_canvas(data["marcos"]))
    card_layout.addWidget(_label("Ejes: X rojo · Y verde · Z azul. Longitud de eje normalizada.", "hint"))
    layout.addWidget(card)
    layout.addStretch(1)
    return _page(body)


def _dh_tab(data: dict) -> QWidget:
    rows = data["tabla_dh"]
    table = QTableWidget(len(rows), len(DH_HEADERS))
    table.setObjectName("dhTable")
    table.setHorizontalHeaderLabels(DH_HEADERS)
    table.verticalHeader().setVisible(False)
    for r, row in enumerate(rows):
        is_theta = row["variable"] == "theta"
        var_col = 4 if is_theta else 3  # columna de la variable articular (θ o d)
        cells = [
            str(row["i"]), _num(row["alpha_prev"]), _num(row["a_prev"]),
            _num(row["d"]), _num(row["theta"]), row["tipo"],
            f"{'θ' if is_theta else 'd'}{row['i']}",
        ]
        for c, text in enumerate(cells):
            item = _cell(text)
            if c in (var_col, 6):
                item.setForeground(QColor(ACCENT))
                item.setBackground(QColor(ACCENT_BG))
            elif 1 <= c <= 4:
                item.setForeground(QColor(MUTED))
            table.setItem(r, c, item)
    _readonly(table)

    card, layout = _card("Tabla de parámetros DH")
    layout.addWidget(table)
    layout.addWidget(_label(DH_NOTE, "hint"))
    content = QWidget()
    QVBoxLayout(content).addWidget(card)
    return _page(content)


def _transformations_tab(data: dict) -> QWidget:
    content = QWidget()
    grid = QGridLayout(content)
    for k, m in enumerate(data["matrices"]):
        grid.addWidget(_matrix_card(m["label"], m["matriz"]), k // 2, k % 2)
    grid.setRowStretch(len(data["matrices"]) // 2 + 1, 1)
    return _page(content)


def _fk_tab(data: dict) -> QWidget:
    fk = data["fk"]
    body = QWidget()
    layout = QVBoxLayout(body)

    top = QHBoxLayout()
    pos_card, pos_layout = _card("Posición p [m]")
    pos_layout.addWidget(_label("\n".join(f"{k} = {v:+.4f} m" for k, v in zip("xyz", fk["p"])), "readout"))
    top.addWidget(pos_card)
    rpy_card, rpy_layout = _card("Orientación (roll, pitch, yaw)")
    rpy_layout.addWidget(_label("\n".join(
        f"{k} = {v:+.2f}°" for k, v in zip(("roll", "pitch", "yaw"), fk["rpy_deg"])), "readout"))
    top.addWidget(rpy_card)
    layout.addLayout(top)

    mid = QHBoxLayout()
    mid.addWidget(_matrix_card("Matriz de rotación R (3×3)", fk["R"]))
    mid.addWidget(_matrix_card("Transformación T0n (4×4)", fk["T0n"]))
    layout.addLayout(mid)
    layout.addStretch(1)
    return _page(body)


def _jacobian_tab(data: dict) -> QWidget:
    jac = data["jacobiano"]
    cols = jac["columnas"]
    body = QWidget()
    layout = QVBoxLayout(body)

    card, card_layout = _card("Columnas por articulación")
    table = QTableWidget(len(cols), 6)
    table.setHorizontalHeaderLabels(["i", "tipo", "z", "o", "Jv", "Jω"])
    table.verticalHeader().setVisible(False)
    for r, c in enumerate(cols):
        values = [str(c["i"]), c["tipo"], _vec(c["z"]), _vec(c["o"]), _vec(c["Jv"]), _vec(c["Jw"])]
        for k, text in enumerate(values):
            table.setItem(r, k, _cell(text))
    card_layout.addWidget(_readonly(table))
    card_layout.addWidget(_label(JAC_NOTE, "hint"))
    layout.addWidget(card)

    J = jac["J"]
    names = [f"{'θ' if c['tipo'] == 'R' else 'd'}{c['i']}" for c in cols]
    card, card_layout = _card("Jacobiano geométrico J (6×n)")
    jt = QTableWidget(len(J), len(J[0]) if J else 0)
    jt.setHorizontalHeaderLabels(names)
    jt.setVerticalHeaderLabels(JAC_ROWS)
    for r, row in enumerate(J):
        for k, v in enumerate(row):
            jt.setItem(r, k, _cell(_num(v)))
    card_layout.addWidget(_readonly(jt))
    layout.addWidget(card)
    layout.addStretch(1)
    return _page(body)


def _singularity_tab(data: dict) -> QWidget:
    sing = data["singularidad"]
    body = QWidget()
    layout = QVBoxLayout(body)

    head = QHBoxLayout()
    head.addWidget(_badge(sing["estado"], "error" if sing["estado"] == "SINGULAR" else "ok"))
    head.addWidget(_badge(f"Rango {sing['rango']}", "info"))
    head.addStretch(1)
    layout.addLayout(head)

    card, card_layout = _card("Valores singulares de J")
    card_layout.addWidget(_sv_canvas(sing["valores_singulares"]))
    card_layout.addWidget(_label(sing["mensaje"], "cardText"))
    layout.addWidget(card)
    layout.addStretch(1)
    return _page(body)


def _check_label(ok: bool | None, text: str) -> QLabel:
    state = "muted" if ok is None else ("ok" if ok else "error")
    mark = {"ok": "✔", "error": "✘", "muted": "–"}[state]
    lbl = _label(f"{mark}  {text}", "check")
    lbl.setProperty("state", state)
    return lbl


def _verification_tab(data: dict) -> QWidget:
    ver = data["verificacion"]
    err = ver["error_ik_fk"]
    checks = [
        (bool(ver["dimensiones_ok"]), "Dimensiones correctas"),
        (ver["ortogonalidad"] < TOL, f"R ortogonal (error máx. {ver['ortogonalidad']:.2e})"),
        (abs(ver["det_R"] - 1.0) < TOL, f"det R = +1 (valor {ver['det_R']:.6f})"),
        (None if err is None else err < TOL_IK,
         "Error IK → FK no aplica (sin IK)" if err is None else f"Error IK → FK = {err:.2e} m"),
    ]
    passed = all(ok for ok, _ in checks if ok is not None)
    body = QWidget()
    layout = QVBoxLayout(body)
    layout.addWidget(_badge("VERIFICADO" if passed else "REVISAR", "ok" if passed else "error"))

    card, card_layout = _card("Comprobaciones")
    for ok, text in checks:
        card_layout.addWidget(_check_label(ok, text))
    layout.addWidget(card)
    layout.addStretch(1)
    return _page(body)


TABS = (
    ("Robot", _robot_tab),
    ("Tabla DH", _dh_tab),
    ("Transformaciones", _transformations_tab),
    ("Cinemática directa", _fk_tab),
    ("Jacobiano", _jacobian_tab),
    ("Singularidades", _singularity_tab),
    ("Verificación", _verification_tab),
)


class ExamTabs(QTabWidget):
    """Muestra el análisis de examen en una pestaña por sección."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("examTabs")
        self.setDocumentMode(True)

    def set_analysis(self, data: dict) -> None:
        """Reconstruye todas las pestañas con la respuesta de /api/exam/analyze."""
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            self.show_error("Respuesta incompleta del análisis; faltan: " + ", ".join(missing))
            return
        try:
            pages = [(title, builder(data)) for title, builder in TABS]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self.show_error(f"Formato inesperado en la respuesta: {exc}")
            return
        self._clear()
        for title, page in pages:
            self.addTab(page, title)

    def show_error(self, message: str) -> None:
        """Sustituye las pestañas por un aviso de error."""
        self._clear()
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(_label(message, "warning"))
        layout.addStretch(1)
        self.addTab(page, "Error")

    def _clear(self) -> None:
        while self.count():
            page = self.widget(0)
            self.removeTab(0)
            page.deleteLater()
