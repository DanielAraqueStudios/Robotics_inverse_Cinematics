"""Traza paso a paso de la solución numérica de cinemática inversa (salida en español)."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .dh import forward_kinematics
from .ik import _joint_bounds, _pose_error, _reach_bound, _residual, solve_ik
from .jacobian import geometric_jacobian
from .models import Robot
from .singularity import analyze_position_singularity
from .steps import _paso, matrix_lines

MOSTRAR_PRIMERAS = 10
MOSTRAR_ULTIMAS = 3


def numeric_ik_steps(robot: Robot, target_T, seed=None, max_iter: int = 200) -> list[dict]:
    """Derivación de una ejecución de mínimos cuadrados, con cada evaluación registrada."""
    T = np.asarray(target_T, dtype=float)
    pasos: list[dict] = []
    roll, pitch, yaw = Rotation.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=True)[::-1]

    pasos.append(_paso("1. Pose objetivo", [
        f"posición p = [{T[0, 3]:+.4f}, {T[1, 3]:+.4f}, {T[2, 3]:+.4f}] m",
        f"orientación: roll = {roll:+.4f}°, pitch = {pitch:+.4f}°, yaw = {yaw:+.4f}°",
        "Matriz homogénea objetivo:",
    ] + matrix_lines(T)))

    lb, ub = _joint_bounds(robot)
    alcance = _reach_bound(robot, lb, ub)
    distancia = float(np.linalg.norm(T[:3, 3]))
    pasos.append(_paso("2. Verificación de alcance", [
        f"|p| = {distancia:.4f} m,  cota de alcance (suma |a| + |d|) = {alcance:.4f} m",
        "alcanzable" if distancia <= alcance + 1e-6
        else "INALCANZABLE: |p| supera la cota de alcance; el solucionador se detiene aquí",
    ]))

    if seed is None:
        x0 = np.clip(np.zeros(robot.n_dof), lb, ub)
        nota = "configuración cero (recortada a los límites articulares)"
    else:
        x0 = np.clip(np.asarray(seed, dtype=float), lb, ub)
        nota = "semilla proporcionada (recortada a los límites articulares)"
    pasos.append(_paso("3. Estimación inicial", [
        f"inicio q0 = [{', '.join(f'{v:+.4f}' for v in x0)}]  ({nota})",
        "Límites articulares:",
    ] + [f"  q{i + 1}: [{lo:+.4f}, {hi:+.4f}]" for i, (lo, hi) in enumerate(zip(lb, ub))]))

    historial: list[tuple[float, float]] = []

    def trazado(q):
        r = _residual(q, robot, T)
        historial.append((float(np.linalg.norm(r[:3])), float(np.linalg.norm(r[3:]))))
        return r

    sol = least_squares(
        trazado, x0, bounds=(lb, ub), method="trf", max_nfev=max_iter,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )

    filas = ["  evaluación   |error posición| [m]   |error rotación| [rad]"]
    numeradas = list(enumerate(historial, 1))
    if len(numeradas) > MOSTRAR_PRIMERAS + MOSTRAR_ULTIMAS:
        vista = numeradas[:MOSTRAR_PRIMERAS] + [None] + numeradas[-MOSTRAR_ULTIMAS:]
    else:
        vista = numeradas
    for item in vista:
        if item is None:
            filas.append("   ...")
            continue
        k, (pe, re_) = item
        filas.append(f"  {k:>10d}   {pe:>18.3e}   {re_:>20.3e}")
    pasos.append(_paso("4. Iteraciones de mínimos cuadrados (reflectivo de región de confianza)", [
        "Residuo r(q) = [error de posición (3); vector de rotación de R_objetivo · R_actual^T (3)]",
        f"evaluaciones de residuo totales: {len(historial)}  (incluye el Jacobiano por diferencias finitas)",
    ] + filas + [f"estado del solucionador: {sol.message}"]))

    q = sol.x
    error_pose = _pose_error(T, forward_kinematics(robot, q))
    error_pos = float(np.linalg.norm(error_pose[:3]))
    error_rot = float(np.linalg.norm(error_pose[3:]))
    pasos.append(_paso("5. Solución y error", [
        "q = [" + ", ".join(f"{math.degrees(v) if j.kind == 'R' else v:+.4f} {'°' if j.kind == 'R' else 'm'}"
                            for v, j in zip(q, robot.joints)) + "]",
        f"error final de posición = {error_pos:.3e} m,  error final de rotación = {error_rot:.3e} rad",
    ]))

    J = geometric_jacobian(robot, q)
    sing = analyze_position_singularity(J, robot)
    estado = "SINGULAR" if sing["is_singular"] else "NO SINGULAR"
    pasos.append(_paso("6. Verificación de singularidad en la solución", [
        f"valores singulares de Jv: [{', '.join(f'{s:.4f}' for s in sing['singular_values'])}]",
        f"estado: {estado}",
    ]))

    resultado = solve_ik(robot, T, seeds=None if seed is None else [seed])
    motivos = {
        "ok": "ok",
        "unreachable": "inalcanzable",
        "max_iter": "máximo de iteraciones",
        "out_of_limits": "fuera de límites",
        "singular": "singular",
    }
    pasos.append(_paso("7. Resultado final", [
        f"motivo: {motivos.get(resultado.reason, resultado.reason)},  convergió: {'sí' if resultado.converged else 'no'}",
        f"soluciones encontradas con todas las semillas: {len(resultado.solutions)}",
    ]))
    return pasos
