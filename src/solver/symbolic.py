"""Cinemática simbólica con parámetros DH (sympy).

Convención (DH estándar, igual que solver.dh):
    T_i-1,i = Rz(theta_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)
"""

from __future__ import annotations

import math

import sympy as sp

from solver.models import Joint, Robot


def _exact(value: float):
    """Convierte un float a expresión exacta cuando es un múltiplo simple de pi o un entero."""
    if value == 0:
        return sp.Integer(0)
    for den in (1, 2, 3, 4, 6, 8, 12):
        k = round(value * den / math.pi)
        if k != 0 and abs(value - k * math.pi / den) < 1e-12:
            return sp.Rational(k, den) * sp.pi
    if abs(value - round(value)) < 1e-12:
        return sp.Integer(round(value))
    return sp.Float(value, 15)


def _variable_symbol(index: int) -> sp.Symbol:
    return sp.Symbol(f"q{index}")


def _joint_params(joint: Joint, index: int):
    """Devuelve (theta, d, a, alpha, variable) de una articulación con su símbolo qi."""
    qi = _variable_symbol(index)
    if joint.kind == "R":
        theta = _exact(joint.theta) + qi
        d = _exact(joint.d)
        variable = "theta"
    else:
        theta = _exact(joint.theta)
        d = _exact(joint.d) + qi
        variable = "d"
    return theta, d, _exact(joint.a), _exact(joint.alpha), variable


def dh_table(robot: Robot) -> list[dict]:
    """Tabla DH simbólica: una fila por articulación."""
    rows = []
    for i, joint in enumerate(robot.joints, start=1):
        theta, d, a, alpha, variable = _joint_params(joint, i)
        rows.append(
            {
                "i": i,
                "theta_i": theta,
                "d_i": d,
                "a_i": a,
                "alpha_i": alpha,
                "tipo": joint.kind,
                "variable": variable,
                "simbolo": _variable_symbol(i),
            }
        )
    return rows


def symbolic_dh_matrix(theta, d, a, alpha) -> sp.Matrix:
    """Matriz homogénea 4x4 simbólica de un eslabón: Rz(theta)·Tz(d)·Tx(a)·Rx(alpha)."""
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


def _link_matrices(robot: Robot) -> list[sp.Matrix]:
    return [
        symbolic_dh_matrix(row["theta_i"], row["d_i"], row["a_i"], row["alpha_i"])
        for row in dh_table(robot)
    ]


def symbolic_chain(robot: Robot) -> list[sp.Matrix]:
    """Transformaciones acumuladas simplificadas T0,1 ... T0,n."""
    acc = sp.eye(4)
    out = []
    for link in _link_matrices(robot):
        acc = (acc * link).applyfunc(sp.simplify)
        out.append(acc)
    return out


def multiplication_steps(robot: Robot) -> list[dict]:
    """Pasos de la cadena de productos T0,k = T0,k-1 * T(k-1),k con su resultado simplificado.

    Los nombres usan la forma T01, T12, ... (válida para robots de hasta 9 articulaciones).
    """
    links = _link_matrices(robot)
    chain = symbolic_chain(robot)
    steps = []
    for k in range(1, len(links) + 1):
        if k == 1:
            operacion = "T01"
            expresion = sp.simplify(links[0])
        else:
            operacion = f"T0{k - 1}*T{k - 1}{k}"
            expresion = chain[k - 1]
        steps.append(
            {
                "paso": k,
                "operacion": operacion,
                "resultado": f"T0{k}",
                "texto": sp.sstr(expresion),
                "matriz": expresion,
            }
        )
    return steps
