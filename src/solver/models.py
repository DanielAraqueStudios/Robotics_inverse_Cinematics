"""Modelo de robot basado en parámetros Denavit-Hartenberg (convención estándar)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

JointKind = Literal["R", "P"]


@dataclass(frozen=True)
class Joint:
    """Una articulación con sus parámetros DH.

    kind:  "R" rotacional (la variable es theta) o "P" prismática (la variable es d).
    a, alpha: constantes del eslabón.
    theta: offset constante (solo se usa si kind == "R").
    d: offset constante (solo se usa si kind == "P").
    lower/upper: límites de la variable articular (None = sin límite).
    """

    kind: JointKind
    a: float = 0.0
    alpha: float = 0.0
    theta: float = 0.0
    d: float = 0.0
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass(frozen=True)
class Robot:
    name: str
    joints: tuple[Joint, ...] = field(default_factory=tuple)

    @property
    def n_dof(self) -> int:
        return len(self.joints)

    @property
    def kinds(self) -> str:
        return "".join(j.kind for j in self.joints)
