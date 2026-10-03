"""Marcos de referencia, tabla DH y descripción en español para ambas convenciones.

Convención modificada (Craig): el eje de la articulación i es z_i (marco i).
Convención estándar: el eje de la articulación i es z_{i-1} (marco i-1).
"""

from __future__ import annotations

import numpy as np

from .dh import _check_q, chain_transforms, link_matrices
from .mdh import chain_transforms_mdh, joint_params_mdh, link_matrices_mdh, validate_convention
from .models import Robot


def link_transforms(robot: Robot, q, convention: str = "modified") -> list[np.ndarray]:
    """Matrices T_{i-1,i} de cada eslabón para la convención dada."""
    validate_convention(convention)
    if convention == "modified":
        return link_matrices_mdh(robot, q)
    return link_matrices(robot, q)


def frame_transforms(robot: Robot, q, convention: str = "modified") -> list[np.ndarray]:
    """Marcos T_0,0 (identidad) ... T_0,n para la convención dada."""
    validate_convention(convention)
    chain = chain_transforms_mdh(robot, q) if convention == "modified" else chain_transforms(robot, q)
    return [np.eye(4)] + chain


def joint_frame_index(i: int, convention: str = "modified") -> int:
    """Índice k del marco cuyo eje z_k es el eje de la articulación i (1-based)."""
    return i if convention == "modified" else i - 1


def describe_frames(robot: Robot, q, convention: str = "modified") -> list[dict]:
    """Un marco por índice i = 0..n con origen y ejes en coordenadas de la base.

    joint_axis es None en el marco 0 (no hay articulación 0). Para la articulación i
    se da el eje z_k de joint_frame_index(i, convention).
    """
    frames = frame_transforms(robot, q, convention)
    _check_q(robot, q)
    marcos = []
    for i, T in enumerate(frames):
        entry = {
            "name": f"{{{i}}}",
            "origin": T[:3, 3].tolist(),
            "x_axis": T[:3, 0].tolist(),
            "y_axis": T[:3, 1].tolist(),
            "z_axis": T[:3, 2].tolist(),
            "joint_axis": None,
        }
        if i == 0:
            entry["descripcion"] = "Marco base {0}: referencia fija del robot."
        else:
            joint = robot.joints[i - 1]
            k = joint_frame_index(i, convention)
            entry["joint_axis"] = frames[k][:3, 2].tolist()
            movimiento = "rotacional alrededor de" if joint.kind == "R" else "prismático a lo largo de"
            entry["descripcion"] = f"Eje de la articulación {i}: {movimiento} z_{k}"
        marcos.append(entry)
    return marcos


def dh_table_mdh(robot: Robot, q) -> list[dict]:
    """Filas de la tabla DH modificada con la variable articular sustituida.

    alpha_prev y a_prev son alpha_{i-1} y a_{i-1} (Craig); d y theta son los valores de la fila.
    """
    q = _check_q(robot, q)
    rows = []
    for i, (joint, qi) in enumerate(zip(robot.joints, q), 1):
        theta, d = joint_params_mdh(joint, qi)
        rotacional = joint.kind == "R"
        rows.append({
            "i": i,
            "alpha_prev": joint.alpha,
            "a_prev": joint.a,
            "d": d,
            "theta": theta,
            "tipo": joint.kind,
            "variable": "theta" if rotacional else "d",
            "constantes": ["alpha_{i-1}", "a_{i-1}", "d_i" if rotacional else "theta_i"],
            "variables": ["theta_i"] if rotacional else ["d_i"],
        })
    return rows


def dh_table_estandar(robot: Robot, q) -> list[dict]:
    """Filas de la tabla DH estándar con la variable articular sustituida."""
    q = _check_q(robot, q)
    rows = []
    for i, (joint, qi) in enumerate(zip(robot.joints, q), 1):
        rotacional = joint.kind == "R"
        theta = joint.theta + qi if rotacional else joint.theta
        d = joint.d if rotacional else joint.d + qi
        rows.append({
            "i": i,
            "alpha": joint.alpha,
            "a": joint.a,
            "d": d,
            "theta": theta,
            "tipo": joint.kind,
            "variable": "theta" if rotacional else "d",
            "constantes": ["alpha_i", "a_i", "d_i" if rotacional else "theta_i"],
            "variables": ["theta_i"] if rotacional else ["d_i"],
        })
    return rows


def dh_table(robot: Robot, q, convention: str = "modified") -> list[dict]:
    """Tabla DH para la convención dada."""
    validate_convention(convention)
    return dh_table_mdh(robot, q) if convention == "modified" else dh_table_estandar(robot, q)
