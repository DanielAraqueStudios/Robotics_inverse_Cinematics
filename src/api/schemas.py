"""Esquemas de entrada y salida de la API (validación en el límite del sistema)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

MAX_JOINTS = 12


class JointIn(BaseModel):
    kind: Literal["R", "P"]
    a: float = 0.0
    alpha: float = 0.0
    theta: float = 0.0
    d: float = 0.0
    lower: Optional[float] = None
    upper: Optional[float] = None


class RobotIn(BaseModel):
    name: str = Field(default="robot", max_length=60)
    joints: List[JointIn] = Field(min_length=1, max_length=MAX_JOINTS)
    tool: Optional[List[float]] = None

    @field_validator("tool")
    @classmethod
    def check_tool(cls, tool: Optional[List[float]]) -> Optional[List[float]]:
        if tool is not None and len(tool) != 4:
            raise ValueError("La herramienta debe tener 4 valores: theta, d, a, alpha")
        return tool

    @field_validator("joints")
    @classmethod
    def check_limits(cls, joints: List[JointIn]) -> List[JointIn]:
        for i, j in enumerate(joints, start=1):
            if j.lower is not None and j.upper is not None and j.lower > j.upper:
                raise ValueError(f"Articulación {i}: el límite inferior supera al superior")
        return joints


class FKRequest(BaseModel):
    robot: RobotIn
    q: List[float] = Field(min_length=1, max_length=MAX_JOINTS)


class IKRequest(BaseModel):
    robot: RobotIn
    position: List[float] = Field(min_length=3, max_length=3)
    rpy_deg: List[float] = Field(min_length=3, max_length=3)
    seed: Optional[List[float]] = Field(default=None, max_length=MAX_JOINTS)


class ChainState(BaseModel):
    origins: List[List[float]]
    end_position: List[float]
    rotation: List[List[float]]
    matrix: List[List[float]]
    singularity: str
    singularity_message: str


class IKSolution(BaseModel):
    q: List[float]
    chain: ChainState


class IKResponse(BaseModel):
    converged: bool
    reason: str
    singular_flag: bool
    position_error: Optional[float]
    orientation_error: Optional[float]
    solutions: List[IKSolution]
