"""Convención DH modificada (Craig): T_{i-1,i} = Rx(alpha_{i-1}) · Tx(a_{i-1}) · Rz(theta_i) · Tz(d_i).

En esta convención Joint.alpha y Joint.a de la articulación i se leen como
alpha_{i-1} y a_{i-1} (parámetros del eslabón anterior, notación de Craig).
Las reglas de variable articular son las mismas que en DH estándar:
    rotacional: theta_i = offset + q_i,  d_i constante
    prismática: d_i = offset + q_i,      theta_i constante

tool: (alpha, a) opcional de un marco fijo final, equivalente a la transformación
Rx(alpha_n) · Tx(a_n) del último eslabón. Sin él el efector coincide con el marco n.
"""

from __future__ import annotations

import math

import numpy as np

from .dh import _check_q
from .models import Joint, Robot

CONVENTIONS = ("modified", "standard")


def validate_convention(convention: str) -> str:
    """Devuelve la convención si es válida; si no, lanza ValueError en español."""
    if convention not in CONVENTIONS:
        raise ValueError(f"Convención no válida: '{convention}'; se esperaba 'modified' o 'standard'")
    return convention


def _rot_x(alpha: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, ca, -sa, 0.0],
            [0.0, sa, ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rot_z(theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, 0.0],
            [st, ct, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _trans_x(a: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = a
    return T


def _trans_z(d: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = d
    return T


def mdh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Matriz homogénea 4x4 Rx(alpha) · Tx(a) · Rz(theta) · Tz(d) (alpha y a son alpha_{i-1}, a_{i-1})."""
    return _rot_x(alpha) @ _trans_x(a) @ _rot_z(theta) @ _trans_z(d)


def joint_params_mdh(joint: Joint, qi: float) -> tuple[float, float]:
    """(theta_i, d_i) de una articulación con la variable qi sustituida."""
    if joint.kind == "R":
        return joint.theta + qi, joint.d
    return joint.theta, joint.d + qi


def _tool_matrix_mdh(robot: Robot, tool: tuple[float, float] | None) -> np.ndarray | None:
    """Marco de herramienta en DH modificada.

    Un argumento explícito tool = (alpha, a) conserva su significado antiguo; si es None se
    usa robot.tool (theta, d, a, alpha) cuando existe; si no, no hay marco extra.
    """
    if tool is not None:
        alpha, a = tool
        return mdh_matrix(0.0, 0.0, a, alpha)
    if robot.has_tool:
        return mdh_matrix(*robot.tool)
    return None


def link_matrices_mdh(robot: Robot, q, tool: tuple[float, float] | None = None) -> list[np.ndarray]:
    """Matrices T_{i-1,i} de cada eslabón, seguidas del marco de herramienta si lo hay."""
    q = _check_q(robot, q)
    mats = []
    for joint, qi in zip(robot.joints, q):
        theta, d = joint_params_mdh(joint, qi)
        mats.append(mdh_matrix(theta, d, joint.a, joint.alpha))
    extra = _tool_matrix_mdh(robot, tool)
    if extra is not None:
        mats.append(extra)
    return mats


def chain_transforms_mdh(robot: Robot, q, tool: tuple[float, float] | None = None) -> list[np.ndarray]:
    """Transformaciones acumuladas T_0,1 ... T_0,n (la última es T_0n)."""
    acc = np.eye(4)
    out = []
    for link in link_matrices_mdh(robot, q, tool):
        acc = acc @ link
        out.append(acc)
    return out


def forward_kinematics_mdh(robot: Robot, q, tool: tuple[float, float] | None = None) -> np.ndarray:
    """T_0n(q) en convención DH modificada."""
    return chain_transforms_mdh(robot, q, tool)[-1]


def mdh_to_standard_robot(robot: Robot, tool: tuple[float, float] | None = None) -> Robot:
    """Robot equivalente en DH estándar con las mismas variables articulares.

    Mapeo exacto (requiere alpha_0 = a_0 = 0 en la primera articulación):
        estándar i: theta_i, d_i iguales;  (alpha_i, a_i) = (alpha_{i-1}, a_{i-1}) de la articulación i+1
        estándar n: (alpha_n, a_n) = tool (o 0 si no hay herramienta).
    Si tool es None y robot.tool existe, la articulación n toma (alpha, a) de robot.tool y el
    robot estándar recibe el marco fijo (theta, d, 0, 0) como herramienta.
    """
    base = robot.joints[0]
    if base.alpha != 0.0 or base.a != 0.0:
        raise ValueError("La conversión exacta requiere alpha_0 = a_0 = 0 en la primera articulación")
    n = robot.n_dof
    use_robot_tool = tool is None and robot.has_tool
    joints = []
    for i, joint in enumerate(robot.joints):
        if i + 1 < n:
            nxt = robot.joints[i + 1]
            alpha, a = nxt.alpha, nxt.a
        elif use_robot_tool:
            alpha, a = robot.tool[3], robot.tool[2]
        else:
            alpha, a = tool if tool is not None else (0.0, 0.0)
        joints.append(
            Joint(joint.kind, a=a, alpha=alpha, theta=joint.theta, d=joint.d, lower=joint.lower, upper=joint.upper)
        )
    std_tool = (robot.tool[0], robot.tool[1], 0.0, 0.0) if use_robot_tool else (0.0, 0.0, 0.0, 0.0)
    return Robot(robot.name, tuple(joints), tool=std_tool)
