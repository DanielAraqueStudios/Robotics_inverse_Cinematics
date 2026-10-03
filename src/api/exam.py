"""Endpoint del análisis de examen: convención DH modificada por defecto, o estándar."""

from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from solver.exam import analyze_exam
from solver.models import Joint, Robot

from .schemas import MAX_JOINTS, RobotIn

router = APIRouter()


class ExamRequest(BaseModel):
    robot: RobotIn
    q: List[float] = Field(min_length=1, max_length=MAX_JOINTS)
    convention: Literal["modified", "standard"] = "modified"


def _to_robot(payload: RobotIn) -> Robot:
    joints = tuple(
        Joint(j.kind, a=j.a, alpha=j.alpha, theta=j.theta, d=j.d, lower=j.lower, upper=j.upper)
        for j in payload.joints
    )
    tool = tuple(payload.tool) if payload.tool is not None else (0.0, 0.0, 0.0, 0.0)
    return Robot(payload.name, joints, tool=tool)


@router.post("/api/exam/analyze")
def exam_analyze(req: ExamRequest) -> dict:
    robot = _to_robot(req.robot)
    if len(req.q) != robot.n_dof:
        raise HTTPException(422, f"Se esperaban {robot.n_dof} variables articulares")
    try:
        return analyze_exam(robot, req.q, req.convention)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
