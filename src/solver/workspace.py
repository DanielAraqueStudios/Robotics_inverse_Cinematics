"""Alcance máximo y muestreo del espacio de trabajo de un robot.

El alcance por eslabones es una cota superior de la distancia desde el origen de la base
hasta el origen del efector final. Muestrear posiciones con cinemática directa da una
estimación del volumen alcanzable, no una prueba de pertenencia exacta.
"""

from __future__ import annotations

import math

import numpy as np

from solver.dh import forward_kinematics
from solver.models import Joint, Robot

from .classify import validate_robot

_DEFAULT_R_LIMITS = (-math.pi, math.pi)
_DEFAULT_P_LIMITS = (-1.0, 1.0)
_REACH_TOL = 1e-9


def _limits(joint: Joint) -> tuple[float, float]:
    """Límites (lo, hi) de la variable articular; None usa el valor por defecto del tipo."""
    default = _DEFAULT_R_LIMITS if joint.kind == "R" else _DEFAULT_P_LIMITS
    lo = default[0] if joint.lower is None else float(joint.lower)
    hi = default[1] if joint.upper is None else float(joint.upper)
    if lo > hi:
        raise ValueError(f"Límites inconsistentes en articulación {joint.kind}: lower {lo} > upper {hi}")
    return lo, hi


def max_reach(robot: Robot) -> float:
    """Cota superior de la distancia desde la base al efector final.

    Por eslabón: |a| + |d_offset|, más para prismáticas el mayor valor absoluto de su
    recorrido (límites; si no hay límite se usa 1.0).
    """
    validate_robot(robot)
    total = 0.0
    for joint in robot.joints:
        total += abs(joint.a) + abs(joint.d)
        if joint.kind == "P":
            lo, hi = _limits(joint)
            total += max(abs(lo), abs(hi))
    total += abs(robot.tool[1]) + abs(robot.tool[2])
    return float(total)


def sample_workspace(robot: Robot, n_samples: int = 5000, rng_seed: int = 0) -> np.ndarray:
    """Posiciones del efector final para variables articulares muestreadas uniformemente.

    Devuelve un array (n, 3) con solo las posiciones finitas. Es reproducible con rng_seed.
    """
    validate_robot(robot)
    if int(n_samples) < 1:
        raise ValueError("n_samples debe ser al menos 1")
    rng = np.random.default_rng(rng_seed)
    bounds = [_limits(j) for j in robot.joints]
    positions = []
    for _ in range(int(n_samples)):
        q = [rng.uniform(lo, hi) for lo, hi in bounds]
        p = forward_kinematics(robot, q)[:3, 3]
        if np.all(np.isfinite(p)):
            positions.append(p)
    if not positions:
        return np.empty((0, 3))
    return np.vstack(positions)


def _check_point(point) -> np.ndarray:
    p = np.asarray(point, dtype=float).reshape(-1)
    if p.size != 3:
        raise ValueError(f"El punto debe tener 3 coordenadas, se recibieron {p.size}")
    if not np.all(np.isfinite(p)):
        raise ValueError("Las coordenadas del punto deben ser finitas")
    return p


def in_workspace_by_reach(robot: Robot, point) -> bool:
    """Indica si el punto está dentro del alcance máximo desde la base.

    Condición necesaria pero no suficiente: un punto dentro del alcance puede no ser
    alcanzable. La pertenencia exacta requiere resolver cinemática inversa (IK).
    """
    p = _check_point(point)
    return bool(np.linalg.norm(p) <= max_reach(robot) + _REACH_TOL)


def check_point(robot: Robot, point, n_samples: int = 5000, rng_seed: int = 0) -> dict:
    """Evalúa un punto por alcance y por el volumen muestreado, explicitando el resultado.

    'in_reach' es la cota por alcance. 'in_sampled_bbox' indica si el punto cae dentro de la
    caja envolvente de las muestras. Fuera de la caja no se concluye que sea inalcanzable.
    """
    p = _check_point(point)
    in_reach = in_workspace_by_reach(robot, p)
    samples = sample_workspace(robot, n_samples, rng_seed)
    in_bbox = False
    if samples.shape[0] > 0:
        in_bbox = bool(np.all(p >= samples.min(axis=0)) and np.all(p <= samples.max(axis=0)))
    if not in_reach:
        note = "Fuera del alcance máximo: no es alcanzable (condición necesaria no cumplida)."
    elif in_bbox:
        note = "Dentro del alcance y de la caja muestreada; la pertenencia exacta requiere IK."
    else:
        note = (
            "Dentro del alcance máximo pero fuera de la caja muestreada; el muestreo no prueba "
            "que sea inalcanzable. La pertenencia exacta requiere IK."
        )
    return {"in_reach": in_reach, "in_sampled_bbox": in_bbox, "note": note}


def workspace_summary(robot: Robot, n_samples: int = 5000, rng_seed: int = 0) -> dict:
    """Resumen del espacio de trabajo: alcance máximo (cota) y estimaciones muestreadas."""
    reach = max_reach(robot)
    samples = sample_workspace(robot, n_samples, rng_seed)
    summary = {
        "max_reach": reach,
        "min_reach_estimate": None,
        "sampled_bbox": None,
        "sampled_max_radius": None,
        "n_samples": int(samples.shape[0]),
        "reach_is_bound_estimate": True,
    }
    if samples.shape[0] > 0:
        radii = np.linalg.norm(samples, axis=1)
        summary["min_reach_estimate"] = float(radii.min())
        summary["sampled_max_radius"] = float(radii.max())
        summary["sampled_bbox"] = {
            "min": samples.min(axis=0).tolist(),
            "max": samples.max(axis=0).tolist(),
        }
    return summary
