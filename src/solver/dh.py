"""Matrices DH y cinemática directa.

Convención (Craig / Spong, DH estándar):
    T_i-1,i = Rz(theta_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)
"""

from __future__ import annotations

import math

import numpy as np

from .models import Robot


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Matriz homogénea 4x4 de un eslabón con parámetros DH."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _check_q(robot: Robot, q) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != robot.n_dof:
        raise ValueError(f"Se esperaban {robot.n_dof} variables articulares, se recibieron {q.size}")
    if not np.all(np.isfinite(q)):
        raise ValueError("Las variables articulares deben ser finitas")
    return q


def link_matrices(robot: Robot, q) -> list[np.ndarray]:
    """Matrices de cada eslabón T_{i-1,i} para los valores articulares q."""
    q = _check_q(robot, q)
    mats = []
    for joint, qi in zip(robot.joints, q):
        if joint.kind == "R":
            mats.append(dh_matrix(joint.theta + qi, joint.d, joint.a, joint.alpha))
        else:
            mats.append(dh_matrix(joint.theta, joint.d + qi, joint.a, joint.alpha))
    if robot.has_tool:
        mats.append(dh_matrix(*robot.tool))
    return mats


def chain_transforms(robot: Robot, q) -> list[np.ndarray]:
    """Transformaciones acumuladas T_0,1 ... T_0,n (la última es T_0n)."""
    acc = np.eye(4)
    out = []
    for link in link_matrices(robot, q):
        acc = acc @ link
        out.append(acc)
    return out


def forward_kinematics(robot: Robot, q) -> np.ndarray:
    """T_0n(q): matriz homogénea del efector final."""
    return chain_transforms(robot, q)[-1]
