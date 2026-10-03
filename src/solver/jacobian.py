"""Jacobianos de un robot serie: geométrico, analítico y simbólico.

Convención de salida: cada Jacobiano es 6 x n con filas [Jv; Jw], donde
    Jv: velocidad lineal del efector final expresada en el sistema base.
    Jw: velocidad angular expresada en el sistema base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .dh import chain_transforms, forward_kinematics
from .models import Robot

FD_STEP = 1e-6


@dataclass(frozen=True)
class JointColumn:
    """Datos de una articulación usados para construir una columna geométrica."""

    index: int
    kind: str
    z: np.ndarray
    origin: np.ndarray


def _frames(robot: Robot, q) -> list[np.ndarray]:
    """Marcos T_0,0 (identidad), T_0,1 ... T_0,n."""
    return [np.eye(4)] + chain_transforms(robot, q)


def joint_columns(robot: Robot, q) -> list[JointColumn]:
    """Eje z_{i-1} y origen o_{i-1} de cada articulación, para reportar paso a paso."""
    frames = _frames(robot, q)
    return [
        JointColumn(
            index=i + 1,
            kind=joint.kind,
            z=frames[i][:3, 2].copy(),
            origin=frames[i][:3, 3].copy(),
        )
        for i, joint in enumerate(robot.joints)
    ]


def geometric_jacobian(robot: Robot, q) -> np.ndarray:
    """Jacobiano geométrico 6 x n.

    Revolutas:  Jv_i = z_{i-1} x (o_n - o_{i-1}),  Jw_i = z_{i-1}
    Prismáticas: Jv_i = z_{i-1},                    Jw_i = 0
    """
    frames = _frames(robot, q)
    p_end = frames[-1][:3, 3]
    n = robot.n_dof
    J = np.zeros((6, n))
    for col in joint_columns(robot, q):
        if col.kind == "R":
            J[:3, col.index - 1] = np.cross(col.z, p_end - col.origin)
            J[3:, col.index - 1] = col.z
        else:
            J[:3, col.index - 1] = col.z
    return J


def _rotation_and_position(robot: Robot, q) -> tuple[np.ndarray, np.ndarray]:
    T = forward_kinematics(robot, q)
    return T[:3, :3], T[:3, 3]


def analytic_jacobian(robot: Robot, q, h: float = FD_STEP) -> np.ndarray:
    """Jacobiano analítico 6 x n.

    Parte lineal: dp/dq por diferencias centrales con paso h.
    Parte angular: vector w tal que [w]x = dR/dq_i · R^T (velocidad angular
    en base), obtenido con diferencias centrales de R(q) y la parte
    antisimétrica de dR·R^T. Representación: w en coordenadas de la base.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    n = robot.n_dof
    R0, _ = _rotation_and_position(robot, q)
    J = np.zeros((6, n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = h
        Rp, pp = _rotation_and_position(robot, q + dq)
        Rm, pm = _rotation_and_position(robot, q - dq)
        J[:3, i] = (pp - pm) / (2.0 * h)
        S = ((Rp - Rm) / (2.0 * h)) @ R0.T
        J[3:, i] = [S[2, 1], S[0, 2], S[1, 0]]
    return J


def _sym_dh(theta, d, a, alpha) -> sp.Matrix:
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )


def _symbolic_frames(robot: Robot) -> tuple[list[sp.Matrix], list[sp.Symbol]]:
    """Marcos simbólicos T_0,0 ... T_0,n con variables q1..qn."""
    qs = sp.symbols(f"q1:{robot.n_dof + 1}")
    frames = [sp.eye(4)]
    T = sp.eye(4)
    for joint, qi in zip(robot.joints, qs):
        theta = sp.Float(joint.theta) + (qi if joint.kind == "R" else 0)
        d = sp.Float(joint.d) + (qi if joint.kind == "P" else 0)
        T = T * _sym_dh(theta, d, sp.Float(joint.a), sp.Float(joint.alpha))
        frames.append(T)
    if robot.has_tool:
        tt, td, ta, tal = (sp.Float(v) for v in robot.tool)
        frames.append(T * _sym_dh(tt, td, ta, tal))
    return frames, list(qs)


def symbolic_position(robot: Robot) -> tuple[sp.Matrix, list[sp.Symbol]]:
    """Posición p(q) del efector final como vector simbólico 3x1, y las variables q."""
    frames, qs = _symbolic_frames(robot)
    return frames[-1][:3, 3], qs


def symbolic_jacobian(robot: Robot) -> sp.Matrix:
    """Jacobiano geométrico 6 x n simbólico en q1..qn (sin evaluar)."""
    frames, _ = _symbolic_frames(robot)
    p_end = frames[-1][:3, 3]
    n = robot.n_dof
    J = sp.zeros(6, n)
    for i, joint in enumerate(robot.joints):
        z = frames[i][:3, 2]
        origin = frames[i][:3, 3]
        if joint.kind == "R":
            J[:3, i] = z.cross(p_end - origin)
            J[3:, i] = z
        else:
            J[:3, i] = z
    return J
