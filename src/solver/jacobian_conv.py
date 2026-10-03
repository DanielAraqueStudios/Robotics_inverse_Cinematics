"""Jacobiano geométrico y rango de posición para la convención elegida.

Revolutas:   Jv_i = z_k x (o_n - o_k),  Jw_i = z_k
Prismáticas: Jv_i = z_k,                Jw_i = 0
donde k = joint_frame_index(i, convention).
"""

from __future__ import annotations

import numpy as np

from .frames import frame_transforms, joint_frame_index
from .models import Robot


def jacobian_detail(robot: Robot, q, convention: str = "modified") -> list[dict]:
    """Columnas del Jacobiano geométrico con eje z, origen o, Jv y Jw (arrays numpy)."""
    frames = frame_transforms(robot, q, convention)
    p_end = frames[-1][:3, 3]
    columnas = []
    for i, joint in enumerate(robot.joints, 1):
        k = joint_frame_index(i, convention)
        z = frames[k][:3, 2].copy()
        o = frames[k][:3, 3].copy()
        if joint.kind == "R":
            jv, jw = np.cross(z, p_end - o), z
        else:
            jv, jw = z, np.zeros(3)
        columnas.append({"i": i, "tipo": joint.kind, "z": z, "o": o, "Jv": jv, "Jw": jw})
    return columnas


def geometric_jacobian_conv(robot: Robot, q, convention: str = "modified") -> np.ndarray:
    """Jacobiano geométrico 6 x n con filas [Jv; Jw]."""
    columnas = jacobian_detail(robot, q, convention)
    return np.column_stack([np.concatenate([c["Jv"], c["Jw"]]) for c in columnas])


def generic_position_rank_conv(robot: Robot, convention: str = "modified", n_samples: int = 30, rng_seed: int = 0) -> int:
    """Rango máximo de Jv observado en configuraciones aleatorias (rango genérico del brazo)."""
    rng = np.random.default_rng(rng_seed)
    best = 0
    for _ in range(n_samples):
        q = []
        for j in robot.joints:
            lo = j.lower if j.lower is not None else (-np.pi if j.kind == "R" else -1.0)
            hi = j.upper if j.upper is not None else (np.pi if j.kind == "R" else 1.0)
            q.append(rng.uniform(lo, hi))
        jv = geometric_jacobian_conv(robot, q, convention)[:3]
        best = max(best, int(np.linalg.matrix_rank(jv, tol=1e-9)))
        if best == 3:
            break
    return best
