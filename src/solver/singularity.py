"""Análisis de singularidades a partir de un Jacobiano (SVD)."""

from __future__ import annotations

import numpy as np

from .jacobian import geometric_jacobian
from .models import Robot


def analyze_singularity(J, tol: float = 1e-9, required_rank: int | None = None) -> dict:
    """Analiza un Jacobiano m x n mediante descomposición en valores singulares.

    El umbral de rango es relativo: sv > tol * max(1, sigma_max).
    required_rank: rango que debe tener el Jacobiano en una configuración no
    singular (por defecto min(m, n)).

    Devuelve un diccionario con:
        rank: rango numérico.
        singular_values: valores singulares en orden descendente.
        min_singular_value: menor valor singular.
        is_singular: True si rank < required_rank.
        status: "SINGULAR" o "NO_SINGULAR" (texto explícito para reportes).
        mensaje: descripción en español del estado.
        determinant: det(J) si J es cuadrada, si no None.
        null_space: columnas base del espacio nulo de J (movimientos articulares
            sin efecto en la tarea), shape n x k.
        lost_motion: columnas base del espacio nulo izquierdo de J (direcciones
            de tarea que no pueden generarse), shape m x l.
    """
    J = np.atleast_2d(np.asarray(J, dtype=float))
    m, n = J.shape
    U, s, Vt = np.linalg.svd(J, full_matrices=True)

    threshold = tol * max(1.0, s[0] if s.size else 0.0)
    rank = int(np.sum(s > threshold))
    needed = min(m, n) if required_rank is None else required_rank
    is_singular = rank < needed

    if is_singular:
        status = "SINGULAR"
        mensaje = (
            f"Configuración SINGULAR: rango {rank} < {needed}. "
            f"Se pierden {needed - rank} grado(s) de movilidad."
        )
    else:
        status = "NO_SINGULAR"
        mensaje = f"Configuración no singular: rango {rank} (requerido {needed})."

    return {
        "rank": rank,
        "singular_values": s,
        "min_singular_value": float(s[-1]) if s.size else 0.0,
        "is_singular": is_singular,
        "status": status,
        "mensaje": mensaje,
        "determinant": float(np.linalg.det(J)) if m == n else None,
        "null_space": Vt[rank:].T,
        "lost_motion": U[:, rank:],
    }


def generic_position_rank(robot: Robot, n_samples: int = 30, rng_seed: int = 0) -> int:
    """Rango máximo de Jv observado en configuraciones aleatorias.

    Es el rango que el brazo alcanza en general: 2 para un plano, 3 para un
    espacial. Una configuración con rango menor es singular respecto a esa base.
    """
    rng = np.random.default_rng(rng_seed)
    best = 0
    for _ in range(n_samples):
        q = []
        for j in robot.joints:
            lo = j.lower if j.lower is not None else (-np.pi if j.kind == "R" else -1.0)
            hi = j.upper if j.upper is not None else (np.pi if j.kind == "R" else 1.0)
            q.append(rng.uniform(lo, hi))
        jv = geometric_jacobian(robot, q)[:3]
        best = max(best, int(np.linalg.matrix_rank(jv, tol=1e-9)))
        if best == 3:
            break
    return best


def analyze_position_singularity(J, robot: Robot | None = None, tol: float = 1e-4) -> dict:
    """Singularidad de posición: rango de Jv frente al rango genérico del robot.

    Un brazo plano 2R en extensión completa tiene Jv de rango 1 frente a un
    rango genérico de 2: es singular aunque el Jacobiano completo conserve
    rango 2 gracias a las filas angulares. Si robot es None se usa min(3, n).
    """
    J = np.atleast_2d(np.asarray(J, dtype=float))
    required = generic_position_rank(robot) if robot is not None else None
    return analyze_singularity(J[:3], tol=tol, required_rank=required)
