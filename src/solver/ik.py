"""Cinemática inversa: solución analítica planar 3R y solver numérico general."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .dh import forward_kinematics
from .models import Robot

REACH_TOL = 1e-9
SINGULAR_EPS = 1e-9
SINGULAR_SV_REL = 1e-4
DEDUP_TOL = 1e-5
FD_STEP = 1e-7
DEFAULT_R_LIMIT = math.pi
DEFAULT_P_LIMIT = 1.0


@dataclass
class IKResult:
    """Resultado de solve_ik. reason es 'ok' o la causa del fallo."""

    solutions: list[np.ndarray] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    position_error: float = math.inf
    orientation_error: float = math.inf
    singular_flag: bool = False
    reason: str = "unreachable"


def solve_planar_rrr(
    a1: float, a2: float, a3: float, x: float, y: float, phi: float
) -> tuple[list[np.ndarray], bool]:
    """Cinemática inversa cerrada de un planar 3R (alpha = 0, d = 0).

    Devuelve (soluciones, singular). Cada solución es [theta1, theta2, theta3].
    Sin solución alcanzable devuelve ([], False). Si sin(theta2) ~ 0 devuelve
    una única solución y singular=True en lugar de soluciones duplicadas.
    """
    if a1 == 0 or a2 == 0:
        raise ValueError("Los eslabones a1 y a2 deben ser distintos de cero")

    xw = x - a3 * math.cos(phi)
    yw = y - a3 * math.sin(phi)
    c2 = (xw * xw + yw * yw - a1 * a1 - a2 * a2) / (2.0 * a1 * a2)
    if abs(c2) > 1.0 + REACH_TOL:
        return [], False
    c2 = min(1.0, max(-1.0, c2))

    t2_abs = math.acos(c2)
    if abs(math.sin(t2_abs)) < SINGULAR_EPS:
        elbow_angles, singular = [t2_abs], True
    else:
        elbow_angles, singular = [t2_abs, -t2_abs], False

    solutions = []
    for t2 in elbow_angles:
        t1 = math.atan2(yw, xw) - math.atan2(a2 * math.sin(t2), a1 + a2 * math.cos(t2))
        t3 = phi - t1 - t2
        solutions.append(np.array([t1, t2, t3]))
    return solutions, singular


def _joint_bounds(robot: Robot) -> tuple[np.ndarray, np.ndarray]:
    lb, ub = [], []
    for joint in robot.joints:
        default = DEFAULT_R_LIMIT if joint.kind == "R" else DEFAULT_P_LIMIT
        lb.append(-default if joint.lower is None else joint.lower)
        ub.append(default if joint.upper is None else joint.upper)
    return np.array(lb, dtype=float), np.array(ub, dtype=float)


def _reach_bound(robot: Robot, lb: np.ndarray, ub: np.ndarray) -> float:
    """Cota superior de la distancia del origen de la base al efector final."""
    bound = 0.0
    for joint, lo, hi in zip(robot.joints, lb, ub):
        bound += abs(joint.a) + abs(joint.d)
        if joint.kind == "P":
            bound += max(abs(lo), abs(hi))
    return bound + abs(robot.tool[1]) + abs(robot.tool[2])


def _check_target(target_T) -> np.ndarray:
    T = np.asarray(target_T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"La pose objetivo debe ser 4x4, se recibió {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError("La pose objetivo debe contener valores finitos")
    return T


def _pose_error(T_target: np.ndarray, T_current: np.ndarray) -> np.ndarray:
    pos = T_target[:3, 3] - T_current[:3, 3]
    rot = Rotation.from_matrix(T_target[:3, :3] @ T_current[:3, :3].T).as_rotvec()
    return np.concatenate([pos, rot])


def _residual(q: np.ndarray, robot: Robot, T_target: np.ndarray) -> np.ndarray:
    return _pose_error(T_target, forward_kinematics(robot, q))


def _build_seeds(
    robot: Robot,
    lb: np.ndarray,
    ub: np.ndarray,
    seeds,
    n_random_seeds: int,
    rng_seed: int,
) -> list[np.ndarray]:
    starts = []
    for seed in seeds or []:
        s = np.asarray(seed, dtype=float).reshape(-1)
        if s.size != robot.n_dof:
            raise ValueError(f"Cada semilla debe tener {robot.n_dof} valores, tiene {s.size}")
        starts.append(np.clip(s, lb, ub))
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_random_seeds):
        starts.append(rng.uniform(lb, ub))
    return starts


def _is_singular(robot: Robot, q: np.ndarray, T_target: np.ndarray) -> bool:
    """Jacobiano numérico de la función residuo en q; sv mínimo casi nulo => singular."""
    n = robot.n_dof
    jac = np.empty((6, n))
    for i in range(n):
        step = np.zeros(n)
        step[i] = FD_STEP
        jac[:, i] = (
            _residual(q + step, robot, T_target) - _residual(q - step, robot, T_target)
        ) / (2.0 * FD_STEP)
    sv = np.linalg.svd(jac, compute_uv=False)
    return sv.min() < SINGULAR_SV_REL * max(sv.max(), SINGULAR_EPS)


def _same_solution(robot: Robot, q: np.ndarray, other: np.ndarray) -> bool:
    diff = q - other
    for i, joint in enumerate(robot.joints):
        if joint.kind == "R":
            diff[i] = (diff[i] + math.pi) % (2.0 * math.pi) - math.pi
    return bool(np.max(np.abs(diff)) < DEDUP_TOL)


def _deduplicate(robot: Robot, candidates: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for q in candidates:
        if not any(_same_solution(robot, q, u) for u in unique):
            unique.append(q)
    return unique


def solve_ik(
    robot: Robot,
    target_T,
    seeds=None,
    max_iter: int = 200,
    tol_pos: float = 1e-6,
    tol_rot: float = 1e-6,
    n_random_seeds: int = 20,
    rng_seed: int = 0,
) -> IKResult:
    """Cinemática inversa numérica por mínimos cuadrados (trf) con límites articulares.

    Se lanza un solver desde cada semilla dada y desde n_random_seeds semillas
    aleatorias dentro de los límites. Las soluciones se deduplican módulo 2*pi
    en las articulaciones rotacionales.
    """
    T = _check_target(target_T)
    lb, ub = _joint_bounds(robot)

    if np.linalg.norm(T[:3, 3]) > _reach_bound(robot, lb, ub) + tol_pos:
        return IKResult(reason="unreachable")

    starts = _build_seeds(robot, lb, ub, seeds, n_random_seeds, rng_seed)
    found: list[np.ndarray] = []
    total_iterations = 0
    best_pos_err, best_rot_err = math.inf, math.inf
    hit_max_iter = False
    blocked_by_limit = False

    for x0 in starts:
        sol = least_squares(
            _residual,
            x0,
            args=(robot, T),
            bounds=(lb, ub),
            method="trf",
            max_nfev=max_iter,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        total_iterations += int(sol.nfev)
        pose_err = _pose_error(T, forward_kinematics(robot, sol.x))
        pos_err = float(np.linalg.norm(pose_err[:3]))
        rot_err = float(np.linalg.norm(pose_err[3:]))

        if pos_err <= tol_pos and rot_err <= tol_rot:
            found.append(sol.x.copy())
            continue
        if pos_err < best_pos_err:
            best_pos_err, best_rot_err = pos_err, rot_err
        hit_max_iter = hit_max_iter or sol.status == 0
        at_limit = np.any(np.isclose(sol.x, lb) | np.isclose(sol.x, ub))
        blocked_by_limit = blocked_by_limit or bool(at_limit)

    solutions = _deduplicate(robot, found)
    if not solutions:
        if blocked_by_limit and np.isfinite(best_pos_err):
            reason = "out_of_limits"
        elif hit_max_iter:
            reason = "max_iter"
        else:
            reason = "unreachable"
        return IKResult(
            iterations=total_iterations,
            position_error=best_pos_err,
            orientation_error=best_rot_err,
            reason=reason,
        )

    singular = any(_is_singular(robot, q, T) for q in solutions)
    first = _pose_error(T, forward_kinematics(robot, solutions[0]))
    return IKResult(
        solutions=solutions,
        converged=True,
        iterations=total_iterations,
        position_error=float(np.linalg.norm(first[:3])),
        orientation_error=float(np.linalg.norm(first[3:])),
        singular_flag=singular,
        reason="ok",
    )
