"""Derivaciones paso a paso con los números sustituidos (salida en español).

Cada paso es un dict {"title": str, "lines": [str]}. Los valores se calculan
con las mismas funciones que usa el solucionador, así que los números mostrados
coinciden exactamente con los resultados.
"""

from __future__ import annotations

import math

import numpy as np

from .dh import link_matrices
from .jacobian import geometric_jacobian
from .models import Robot
from .singularity import analyze_position_singularity, generic_position_rank


def _paso(titulo: str, lineas) -> dict:
    return {"title": titulo, "lines": list(lineas)}


def matrix_lines(M, indent: str = "    ") -> list[str]:
    """Matriz como filas alineadas, 4 decimales."""
    return [indent + "[" + "  ".join(f"{v:+10.4f}" for v in fila) + " ]" for fila in np.asarray(M)]


def _rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    from scipy.spatial.transform import Rotation

    yaw, pitch, roll = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    return float(roll), float(pitch), float(yaw)


def fk_steps(robot: Robot, q) -> list[dict]:
    """Derivación completa de cinemática directa para cualquier cadena DH."""
    q = np.asarray(q, dtype=float).reshape(-1)
    n = robot.n_dof
    pasos: list[dict] = []

    lineas = [f"Cadena '{robot.name}': {n} articulaciones, tipos {robot.kinds}"]
    for i, (j, v) in enumerate(zip(robot.joints, q), 1):
        unidad = "rad" if j.kind == "R" else "m"
        extra = f"  ({math.degrees(v):+.3f} grados)" if j.kind == "R" else ""
        lineas.append(f"q{i} = {v:+.4f} {unidad}{extra}")
    pasos.append(_paso("1. Robot y variables articulares", lineas))

    pasos.append(_paso(
        "2. Convención",
        ["DH estándar (Craig/Spong): T(i-1,i) = Rz(theta_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)",
         "Articulación rotacional: theta_i = offset + q_i,  d_i constante",
         "Articulación prismática: d_i = offset + q_i,  theta_i constante"],
    ))

    tabla = ["  i  tipo   theta_i [rad]   d_i [m]    a_i [m]   alpha_i [rad]"]
    valores = []
    for i, (j, v) in enumerate(zip(robot.joints, q), 1):
        theta = j.theta + v if j.kind == "R" else j.theta
        d = j.d + v if j.kind == "P" else j.d
        valores.append((theta, d))
        tabla.append(f"  {i:<2d} {j.kind:<5s} {theta:+13.4f}  {d:+9.4f}  {j.a:+8.4f}  {j.alpha:+12.4f}")
    pasos.append(_paso("3. Tabla DH (con la variable articular sustituida)", tabla))

    mats = link_matrices(robot, q)
    for i, (M, (theta, d), j) in enumerate(zip(mats, valores, robot.joints), 1):
        pasos.append(_paso(
            f"4.{i} Transformación de eslabón T{i - 1}{i}",
            [f"T{i - 1}{i} = Rz({theta:+.4f}) · Tz({d:+.4f}) · Tx({j.a:+.4f}) · Rx({j.alpha:+.4f})"]
            + matrix_lines(M),
        ))

    acc = np.eye(4)
    pasos.append(_paso("5. Producto acumulado, desde el marco base", ["T00 = identidad"] + matrix_lines(acc)))
    for i, M in enumerate(mats, 1):
        acc = acc @ M
        pasos.append(_paso(
            f"5.{i} Producto acumulado T0{i}",
            [f"T0{i} = T0{i - 1} · T{i - 1}{i}", "resultado:"] + matrix_lines(acc),
        ))

    T = acc
    R, p = T[:3, :3], T[:3, 3]
    roll, pitch, yaw = _rpy_deg(R)
    pasos.append(_paso(
        "6. Pose del efector final T0n",
        [f"Posición p = [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}] m",
         "Rotación R:"] + matrix_lines(R)
        + [f"Orientación (Euler ZYX): roll = {roll:+.4f}°, pitch = {pitch:+.4f}°, yaw = {yaw:+.4f}°",
           "Matriz homogénea T0n:"] + matrix_lines(T),
    ))

    ortogonalidad = float(np.max(np.abs(R.T @ R - np.eye(3))))
    pasos.append(_paso(
        "7. Verificación de validez",
        [f"máx |R^T R - I| = {ortogonalidad:.2e}   (debe ser ~0)",
         f"det(R) = {np.linalg.det(R):+.6f}   (debe ser +1)"],
    ))

    J = geometric_jacobian(robot, q)
    sing = analyze_position_singularity(J, robot)
    requerido = generic_position_rank(robot)
    sv = ", ".join(f"{s:.4f}" for s in sing["singular_values"])
    estado = "SINGULAR" if sing["is_singular"] else "NO SINGULAR"
    pasos.append(_paso(
        "8. Verificación de singularidad (Jacobiano de posición)",
        ["Jv = primeras 3 filas del Jacobiano geométrico:"] + matrix_lines(J[:3])
        + [f"valores singulares de Jv: [{sv}]",
           f"rango = {sing['rank']}, rango requerido para este robot = {requerido}",
           f"estado: {estado}"],
    ))
    return pasos


def planar_rrr_steps(a1: float, a2: float, a3: float, x: float, y: float, phi: float) -> tuple[list[dict], list[list[float]]]:
    """Solución cerrada de cinemática inversa para el brazo plano 3R. phi en radianes.

    Devuelve (pasos, soluciones).
    """
    pasos: list[dict] = []
    pasos.append(_paso("1. Datos de entrada", [
        f"longitudes a1 = {a1:.4f}, a2 = {a2:.4f}, a3 = {a3:.4f} m",
        f"objetivo x = {x:+.4f} m, y = {y:+.4f} m, orientación phi = {phi:+.4f} rad ({math.degrees(phi):+.4f}°)",
    ]))
    xw = x - a3 * math.cos(phi)
    yw = y - a3 * math.sin(phi)
    pasos.append(_paso("2. Punto de muñeca (se quita el último eslabón)", [
        "xw = x - a3 · cos(phi),  yw = y - a3 · sin(phi)",
        f"xw = {x:+.4f} - {a3:.4f} · cos({phi:+.4f} rad) = {xw:+.4f}",
        f"yw = {y:+.4f} - {a3:.4f} · sen({phi:+.4f} rad) = {yw:+.4f}",
    ]))
    r2 = xw * xw + yw * yw
    pasos.append(_paso("3. Distancia al cuadrado a la muñeca", [
        "r² = xw² + yw²",
        f"r² = {xw:+.4f}² + {yw:+.4f}² = {r2:.4f}",
    ]))
    c2 = (r2 - a1 * a1 - a2 * a2) / (2 * a1 * a2)
    pasos.append(_paso("4. Ley de cosenos para theta2", [
        "cos(theta2) = (r² - a1² - a2²) / (2 · a1 · a2)",
        f"cos(theta2) = ({r2:.4f} - {a1 * a1:.4f} - {a2 * a2:.4f}) / (2 · {a1:.4f} · {a2:.4f}) = {c2:+.6f}",
    ]))
    if abs(c2) > 1 + 1e-9:
        pasos.append(_paso("Resultado", [f"|cos(theta2)| = {abs(c2):.4f} > 1 : objetivo inalcanzable, sin solución"]))
        return pasos, []

    c2 = max(-1.0, min(1.0, c2))
    t2 = math.acos(c2)
    soluciones: list[list[float]] = []
    ramas = [("theta2 > 0", t2), ("theta2 < 0", -t2)] if t2 > 1e-9 and math.pi - t2 > 1e-9 else [("única", t2)]
    pasos.append(_paso("5. Ramas de la articulación del codo", [
        f"theta2 = ±acos({c2:+.6f}) = ±{math.degrees(t2):.4f}°",
    ] + [f"rama {nombre}: theta2 = {math.degrees(v):+.4f}°" for nombre, v in ramas]))

    for k, (nombre, th2) in enumerate(ramas, 1):
        s2, c2b = math.sin(th2), math.cos(th2)
        k1 = math.atan2(yw, xw)
        k2 = math.atan2(a2 * s2, a1 + a2 * c2b)
        th1 = k1 - k2
        th3 = phi - th1 - th2
        soluciones.append([th1, th2, th3])
        pasos.append(_paso(f"6.{k} Ángulos articulares para la rama {nombre}", [
            f"sen(theta2) = {s2:+.6f},  cos(theta2) = {c2b:+.6f}",
            "theta1 = atan2(yw, xw) - atan2(a2 · sen(theta2), a1 + a2 · cos(theta2))",
            f"atan2(yw, xw) = atan2({yw:+.4f}, {xw:+.4f}) = {math.degrees(k1):+.4f}°",
            f"atan2(a2·sen, a1+a2·cos) = atan2({a2 * s2:+.4f}, {a1 + a2 * c2b:+.4f}) = {math.degrees(k2):+.4f}°",
            f"theta1 = {math.degrees(k1):+.4f} - {math.degrees(k2):+.4f} = {math.degrees(th1):+.4f}°",
            "theta3 = phi - theta1 - theta2",
            f"theta3 = {math.degrees(phi):+.4f} - {math.degrees(th1):+.4f} - {math.degrees(th2):+.4f} = {math.degrees(th3):+.4f}°",
            f"solución: theta = [{math.degrees(th1):+.4f}, {math.degrees(th2):+.4f}, {math.degrees(th3):+.4f}]°",
        ]))
    return pasos, soluciones
