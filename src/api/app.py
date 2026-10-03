"""API FastAPI del solucionador cinemático (FK, IK y estado de la cadena)."""

from __future__ import annotations

import asyncio
import math

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from solver.dh import chain_transforms
from solver.ik import solve_ik
from solver.jacobian import geometric_jacobian
from solver.models import Joint, Robot
from solver.singularity import analyze_position_singularity
from solver.steps import fk_steps, planar_rrr_steps
from solver.steps_numeric import numeric_ik_steps

from .schemas import ChainState, FKRequest, IKRequest, IKResponse, IKSolution, RobotIn

IK_TIMEOUT_S = 5.0

app = FastAPI(title="Cinemática inversa API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


def to_robot(payload: RobotIn) -> Robot:
    joints = tuple(
        Joint(j.kind, a=j.a, alpha=j.alpha, theta=j.theta, d=j.d, lower=j.lower, upper=j.upper)
        for j in payload.joints
    )
    return Robot(payload.name, joints)


def rpy_to_matrix(rpy_deg: list[float]) -> np.ndarray:
    """Matriz de rotación desde ángulos roll, pitch, yaw en grados (ZYX)."""
    r, p, y = (math.radians(v) for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def finite_or_none(value: float) -> float | None:
    """JSON no admite inf/NaN: un error no finito se reporta como null."""
    value = float(value)
    return value if math.isfinite(value) else None


def chain_state(robot: Robot, q) -> ChainState:
    """Estado completo de la cadena: orígenes de cada eslabón y singularidad."""
    frames = chain_transforms(robot, q)
    origins = [[0.0, 0.0, 0.0]] + [T[:3, 3].tolist() for T in frames]
    T = frames[-1]
    sing = analyze_position_singularity(geometric_jacobian(robot, q), robot)
    return ChainState(
        origins=origins,
        end_position=T[:3, 3].tolist(),
        rotation=T[:3, :3].tolist(),
        matrix=T.tolist(),
        singularity=sing["status"],
        singularity_message=sing["mensaje"],
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/fk", response_model=ChainState)
def forward(req: FKRequest) -> ChainState:
    robot = to_robot(req.robot)
    if len(req.q) != robot.n_dof:
        raise HTTPException(422, f"Se esperaban {robot.n_dof} variables articulares")
    try:
        return chain_state(robot, req.q)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/ik", response_model=IKResponse)
async def inverse(req: IKRequest) -> IKResponse:
    robot = to_robot(req.robot)
    target = np.eye(4)
    target[:3, :3] = rpy_to_matrix(req.rpy_deg)
    target[:3, 3] = req.position
    seeds = [req.seed] if req.seed is not None and len(req.seed) == robot.n_dof else None

    try:
        # IK numérico en hilo aparte con tiempo máximo: nunca bloquea el servidor.
        result = await asyncio.wait_for(
            run_in_threadpool(solve_ik, robot, target, seeds),
            timeout=IK_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "El cálculo de cinemática inversa excedió el tiempo máximo") from exc

    solutions = [IKSolution(q=list(map(float, q)), chain=chain_state(robot, q)) for q in result.solutions]
    return IKResponse(
        converged=result.converged,
        reason=result.reason,
        singular_flag=result.singular_flag,
        position_error=finite_or_none(result.position_error),
        orientation_error=finite_or_none(result.orientation_error),
        solutions=solutions,
    )


def is_planar_3r(robot: Robot) -> bool:
    """Planar 3R arm in the form the closed-form solver supports."""
    return (
        robot.n_dof == 3
        and all(j.kind == "R" and j.alpha == 0.0 and j.d == 0.0 and j.theta == 0.0 for j in robot.joints)
        and all(j.a > 0.0 for j in robot.joints)
    )


@app.post("/api/steps/fk")
def fk_steps_endpoint(req: FKRequest) -> dict:
    robot = to_robot(req.robot)
    if len(req.q) != robot.n_dof:
        raise HTTPException(422, f"Se esperaban {robot.n_dof} variables articulares")
    try:
        return {"steps": fk_steps(robot, req.q)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/steps/ik")
async def ik_steps_endpoint(req: IKRequest) -> dict:
    robot = to_robot(req.robot)
    target = np.eye(4)
    target[:3, :3] = rpy_to_matrix(req.rpy_deg)
    target[:3, 3] = req.position
    seed = req.seed if req.seed is not None and len(req.seed) == robot.n_dof else None

    steps: list[dict] = []
    if is_planar_3r(robot) and abs(req.position[2]) < 1e-9 and abs(req.rpy_deg[0]) < 1e-9 and abs(req.rpy_deg[1]) < 1e-9:
        lengths = [j.a for j in robot.joints]
        planar, _ = planar_rrr_steps(*lengths, req.position[0], req.position[1], math.radians(req.rpy_deg[2]))
        steps.append({"title": "PARTE A - Solución cerrada (3R plano)", "lines": []})
        steps.extend(planar)
    try:
        numeric = await asyncio.wait_for(
            run_in_threadpool(numeric_ik_steps, robot, target, seed),
            timeout=IK_TIMEOUT_S,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "El cálculo de pasos excedió el tiempo máximo") from exc
    if steps:
        numeric = [{"title": "PARTE B - Solución numérica (mínimos cuadrados, mismo método que el solucionador)", "lines": []}] + numeric
    steps.extend(numeric)
    return {"steps": steps}
