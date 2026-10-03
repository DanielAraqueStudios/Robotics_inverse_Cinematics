"""Pestaña de solución paso a paso: tarjetas, ecuaciones tipografiadas y matrices en tabla."""

from __future__ import annotations

import html
import re

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout, QWidget

ACCENT = "#4CC9F0"
CARD_BG = "#111827"
EQ_BG = "#0B1220"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
MATRIX_BORDER = "#22D3EE"

_SYMBOLS = [
    (r"\btheta(\d+|_i)", r"θ<sub>\1</sub>"),
    (r"\balpha(\d+|_i)", r"α<sub>\1</sub>"),
    (r"\bphi\b", "φ"),
    (r"\bd_i\b", "d<sub>i</sub>"),
    (r"\ba_i\b", "a<sub>i</sub>"),
    (r"\bd(\d)\b", r"d<sub>\1</sub>"),
    (r"\ba(\d)\b", r"a<sub>\1</sub>"),
    (r"\bq(\d)\b", r"q<sub>\1</sub>"),
    (r"\bx(\d)\b", r"x<sub>\1</sub>"),
    (r"\bxw\b", "x<sub>w</sub>"),
    (r"\byw\b", "y<sub>w</sub>"),
    (r"\bT(\d)(\d)\b", r"T<sup>\1</sup><sub>\2</sub>"),
    (r"\bT0(\d)\b", r"T<sup>0</sup><sub>\1</sub>"),
    (r"\bT00\b", "T<sup>0</sup><sub>0</sub>"),
    (r"\bsen\(", "sen("),
    (r"\bcos\(", "cos("),
    (r"(?<=\d)\s?°", "°"),
    (r"\^2", "²"),
    (r"\bR\^T\b", "Rᵀ"),
    (r"\bdet\(", "det("),
    (r"\bmáx\b", "máx"),
]

_MATRIX_ROW = re.compile(r"^\s{4}\[")
_EQUATION = re.compile(r"[=≤≥]")


def _typeset(escaped: str) -> str:
    out = escaped
    for pattern, repl in _SYMBOLS:
        out = re.sub(pattern, repl, out)
    return out


def _matrix_html(rows: list[str]) -> str:
    cells = []
    for row in rows:
        values = row.strip().strip("[]").split()
        cells.append("<tr>" + "".join(
            f'<td style="padding:2px 9px; text-align:right; font-family:Consolas,monospace; color:{TEXT};">'
            f"{html.escape(v)}</td>" for v in values) + "</tr>")
    return (
        f'<table cellspacing="0" cellpadding="0" style="margin:6px 0 6px 14px; '
        f'border-left:3px solid {MATRIX_BORDER}; border-right:3px solid {MATRIX_BORDER}; '
        f'border-radius:6px;">' + "".join(cells) + "</table>"
    )


def _equation_html(line: str) -> str:
    return (
        f'<div style="margin:4px 0; padding:6px 10px; background:{EQ_BG}; '
        f'border-radius:6px; font-family:Cambria Math,Georgia,serif; font-size:14px; color:{TEXT};">'
        f"{_typeset(html.escape(line))}</div>"
    )


def _text_html(line: str) -> str:
    return f'<div style="margin:2px 0; color:#CBD5E1;">{_typeset(html.escape(line))}</div>'


def _body_html(lines: list[str]) -> str:
    parts: list[str] = []
    matrix: list[str] = []

    def flush():
        if matrix:
            parts.append(_matrix_html(matrix))
            matrix.clear()

    for line in lines:
        if _MATRIX_ROW.match(line):
            matrix.append(line)
            continue
        flush()
        if not line.strip():
            continue
        if _EQUATION.search(line) and not line.startswith(("Posición", "Orientación", "solución", "estado")):
            parts.append(_equation_html(line))
        else:
            parts.append(_text_html(line))
    flush()
    return "".join(parts)


def steps_to_html(steps: list[dict]) -> str:
    cards = []
    for step in steps:
        title = html.escape(step["title"])
        if step["title"].startswith(("PARTE A", "PARTE B")):
            cards.append(
                f'<h2 style="color:{ACCENT}; margin:18px 0 4px 0; font-size:16px;">{title}</h2>'
            )
            continue
        cards.append(
            f'<div style="background:{CARD_BG}; border-left:4px solid {ACCENT}; '
            f'border-radius:8px; padding:10px 14px; margin:8px 0;">'
            f'<div style="color:{ACCENT}; font-weight:700; font-size:14px; margin-bottom:6px;">{title}</div>'
            f"{_body_html(step['lines'])}</div>"
        )
    return (
        f'<html><body style="background:#0F141B; color:{TEXT}; '
        f'font-family:Segoe UI,Arial,sans-serif; font-size:13px;">'
        + "".join(cards)
        + "</body></html>"
    )


def steps_to_plain(steps: list[dict]) -> str:
    out: list[str] = []
    for step in steps:
        out.append("")
        out.append(step["title"])
        out.extend(step["lines"])
    return "\n".join(out).lstrip("\n")


class StepsTab(QWidget):
    """Muestra la derivación completa con formato; el botón copia el texto plano."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plain = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        self.view.setFont(QFont("Segoe UI", 10))
        self.view.setPlaceholderText("Presione «Mostrar solución paso a paso» para ver el desarrollo completo.")
        layout.addWidget(self.view, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        copy_btn = QPushButton("Copiar texto")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self._plain))
        row.addWidget(copy_btn)
        layout.addLayout(row)

    def set_steps(self, steps: list[dict]) -> None:
        self._plain = steps_to_plain(steps)
        self.view.setHtml(steps_to_html(steps))
        self.view.verticalScrollBar().setValue(0)
