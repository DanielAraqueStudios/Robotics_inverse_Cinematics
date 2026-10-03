"""Informe paso a paso en español de la cinemática directa de un robot DH."""

from __future__ import annotations

import sympy as sp

from solver.dh import forward_kinematics
from solver.models import Robot
from solver.symbolic import dh_table, multiplication_steps, symbolic_chain, symbolic_dh_matrix

NO_SOLICITADO = "No solicitado"
CONVENCION = "Rz(theta)·Tz(d)·Tx(a)·Rx(alpha)"


def _round(value) -> float:
    rounded = round(float(value), 4)
    return rounded + 0.0  # evita -0.0


def _fmt(expr) -> str | float:
    """Expresión simbólica como texto; constantes numéricas redondeadas a 4 decimales."""
    expr = sp.sympify(expr)
    if expr.free_symbols:
        return sp.sstr(expr)
    return _round(expr)


def _entrada(robot: Robot, target) -> dict:
    rotacionales = [j for j, joint in enumerate(robot.joints, start=1) if joint.kind == "R"]
    prismaticas = [j for j, joint in enumerate(robot.joints, start=1) if joint.kind == "P"]
    return {
        "nombre": robot.name,
        "tipo_configuracion": robot.kinds,
        "numero_GDL": robot.n_dof,
        "articulaciones_rotacionales": rotacionales or NO_SOLICITADO,
        "articulaciones_prismaticas": prismaticas or NO_SOLICITADO,
        "objetivo": target if target is not None else NO_SOLICITADO,
    }


def _paso_1(robot: Robot) -> list[str]:
    n = robot.n_dof
    return [
        "Asignar el sistema de referencia base {0} en la base del robot.",
        f"Asignar un sistema {{i}} a cada articulación i = 1..{n}, con el eje z_i sobre el eje de la articulación i.",
        "Asignar el sistema {n} en el efector final.",
        "Determinar theta_i, d_i, a_i y alpha_i para cada eslabón según la convención DH estándar.",
    ]


def _paso_2(robot: Robot) -> dict:
    filas = []
    for fila in dh_table(robot):
        filas.append(
            {
                "i": fila["i"],
                "tipo": fila["tipo"],
                "variable": fila["variable"],
                "theta_i": _fmt(fila["theta_i"]),
                "d_i": _fmt(fila["d_i"]),
                "a_i": _fmt(fila["a_i"]),
                "alpha_i": _fmt(fila["alpha_i"]),
            }
        )
    return {"tabla_DH": filas, "convencion": CONVENCION}


def _paso_3(robot: Robot) -> dict:
    individuales = []
    for fila in dh_table(robot):
        mat = symbolic_dh_matrix(fila["theta_i"], fila["d_i"], fila["a_i"], fila["alpha_i"])
        individuales.append({"eslabon": fila["i"], "matriz": sp.sstr(mat.applyfunc(sp.simplify))})
    pasos = [
        {k: v for k, v in paso.items() if k != "matriz"} for paso in multiplication_steps(robot)
    ]
    chain = symbolic_chain(robot)
    return {
        "matrices_individuales": individuales,
        "multiplicacion": pasos,
        "matriz_final_T0n": sp.sstr(chain[-1]),
    }


def _cinematica_directa(robot: Robot, q) -> dict:
    if q is None:
        return {"q": NO_SOLICITADO, "px": NO_SOLICITADO, "py": NO_SOLICITADO, "pz": NO_SOLICITADO, "R": NO_SOLICITADO}
    T = forward_kinematics(robot, q)
    return {
        "q": [_round(v) for v in q],
        "px": _round(T[0, 3]),
        "py": _round(T[1, 3]),
        "pz": _round(T[2, 3]),
        "R": [[_round(v) for v in row] for row in T[:3, :3]],
    }


def _paso_5(robot: Robot) -> dict:
    chain = symbolic_chain(robot)
    T = chain[-1]
    return {
        "px_simbolico": sp.sstr(sp.simplify(T[0, 3])),
        "py_simbolico": sp.sstr(sp.simplify(T[1, 3])),
        "pz_simbolico": sp.sstr(sp.simplify(T[2, 3])),
    }


def _de_resultados(results: dict | None, clave: str):
    if not results or results.get(clave) is None:
        return NO_SOLICITADO
    return results[clave]


def build_report(robot: Robot, q=None, target=None, results=None) -> dict:
    """Construye el informe por pasos. Solo incluye datos calculados o recibidos."""
    return {
        "ENTRADA_PROBLEMA": _entrada(robot, target),
        "PASO_1_SISTEMAS_REFERENCIA": _paso_1(robot),
        "PASO_2_TABLA_DENAVIT_HARTENBERG": _paso_2(robot),
        "PASO_3_MATRICES_TRANSFORMACION": _paso_3(robot),
        "PASO_4_METODO_SOLUCION": {"cinematica_directa": _cinematica_directa(robot, q)},
        "PASO_5_DESARROLLO_MATEMATICO": _paso_5(robot),
        "PASO_6_SINGULARIDADES": _de_resultados(results, "singularity"),
        "PASO_7_ESPACIO_TRABAJO": _de_resultados(results, "workspace"),
        "PASO_8_VERIFICACION_FINAL": _de_resultados(results, "verification"),
        "SALIDA_FINAL_EXAMEN": _de_resultados(results, "resultado_encerrado"),
    }


def _render(valor, nivel: int, lineas: list[str]) -> None:
    sangria = "  " * nivel
    if isinstance(valor, dict):
        for clave, sub in valor.items():
            if isinstance(sub, (dict, list)) and sub:
                lineas.append(f"{sangria}- **{clave}**:")
                _render(sub, nivel + 1, lineas)
            else:
                lineas.append(f"{sangria}- **{clave}**: {sub}")
    elif isinstance(valor, list):
        for item in valor:
            if isinstance(item, (dict, list)):
                lineas.append(f"{sangria}-")
                _render(item, nivel + 1, lineas)
            else:
                lineas.append(f"{sangria}- {item}")
    else:
        lineas.append(f"{sangria}{valor}")


def format_report_text(report: dict) -> str:
    """Texto Markdown en español del informe."""
    lineas = []
    for titulo, contenido in report.items():
        lineas.append(f"## {titulo.replace('_', ' ')}")
        lineas.append("")
        _render(contenido, 0, lineas)
        lineas.append("")
    return "\n".join(lineas)
