"""Análisis completo de un robot para el examen: marcos, tabla DH, FK, Jacobiano, singularidad y verificación."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .classify import classify
from .dh import _check_q
from .frames import describe_frames, dh_table, frame_transforms, link_transforms
from .ik import solve_ik
from .jacobian_conv import generic_position_rank_conv, geometric_jacobian_conv, jacobian_detail
from .mdh import mdh_to_standard_robot, validate_convention
from .models import Robot
from .singularity import analyze_singularity

SING_TOL = 1e-4


def _rpy_deg(R: np.ndarray) -> list[float]:
    yaw, pitch, roll = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    return [float(roll), float(pitch), float(yaw)]


def _matrices(robot: Robot, q, convention: str) -> list[dict]:
    """Eslabones T{i-1}{i} seguidos de los acumulados T0{i}."""
    links = link_transforms(robot, q, convention)
    cumulative = frame_transforms(robot, q, convention)[1:]
    out = [{"label": f"T{i - 1}{i}", "matriz": M.tolist()} for i, M in enumerate(links, 1)]
    out += [{"label": f"T0{i}", "matriz": M.tolist()} for i, M in enumerate(cumulative, 1)]
    return out


def _ik_fk_error(robot: Robot, T0n: np.ndarray, convention: str) -> float | None:
    """Error de posición del primer resultado de IK (semilla cero) frente a la pose de FK.

    La IK se resuelve con la cadena estándar equivalente; el error se mide con la
    cadena de la convención pedida. Devuelve None si no hay solución o no hay equivalencia exacta.
    """
    n = robot.n_dof
    try:
        std = mdh_to_standard_robot(robot) if convention == "modified" else robot
    except ValueError:
        return None
    result = solve_ik(std, T0n, seeds=[np.zeros(n)])
    if not result.solutions:
        return None
    p_sol = frame_transforms(robot, result.solutions[0], convention)[-1][:3, 3]
    return float(np.linalg.norm(p_sol - T0n[:3, 3]))


def _singularity(robot: Robot, J: np.ndarray, convention: str) -> dict:
    required = generic_position_rank_conv(robot, convention)
    sing = analyze_singularity(J[:3], tol=SING_TOL, required_rank=required)
    return {
        "status": sing["status"],
        "rank": sing["rank"],
        "rango_requerido": required,
        "singular_values": [float(s) for s in sing["singular_values"]],
        "mensaje_es": sing["mensaje"],
    }


def analyze_exam(robot: Robot, q, convention: str = "modified") -> dict:
    """Análisis completo del robot en q con la convención DH indicada ('modified' por defecto)."""
    validate_convention(convention)
    q = _check_q(robot, q)
    n = robot.n_dof

    clas = classify(robot)
    T0n = frame_transforms(robot, q, convention)[-1]
    R, p = T0n[:3, :3], T0n[:3, 3]

    columnas = jacobian_detail(robot, q, convention)
    J = geometric_jacobian_conv(robot, q, convention)

    verificacion = {
        "dimensiones_ok": bool(J.shape == (6, n) and len(columnas) == n and T0n.shape == (4, 4)),
        "ortogonalidad": float(np.max(np.abs(R.T @ R - np.eye(3)))),
        "det_R": float(np.linalg.det(R)),
        "ik_fk_error": _ik_fk_error(robot, T0n, convention),
    }

    return {
        "convencion": convention,
        "clasificacion": clas,
        "grados_libertad": n,
        "articulaciones": {"rotacionales": clas["rotational"], "prismaticas": clas["prismatic"]},
        "marcos": describe_frames(robot, q, convention),
        "tabla_dh": dh_table(robot, q, convention),
        "matrices": _matrices(robot, q, convention),
        "fk": {
            "p": p.tolist(),
            "R": R.tolist(),
            "T0n": T0n.tolist(),
            "rpy_deg": _rpy_deg(R),
        },
        "jacobiano": {
            "columnas": [
                {
                    "i": c["i"],
                    "tipo": c["tipo"],
                    "z": c["z"].tolist(),
                    "o": c["o"].tolist(),
                    "Jv": c["Jv"].tolist(),
                    "Jw": c["Jw"].tolist(),
                }
                for c in columnas
            ],
            "J": J.tolist(),
            "Jv": J[:3].tolist(),
            "Jw": J[3:].tolist(),
        },
        "singularidad": _singularity(robot, J, convention),
        "verificacion": verificacion,
    }
