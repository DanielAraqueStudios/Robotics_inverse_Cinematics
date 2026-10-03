"""Clasificación de la configuración de un robot por tipo de articulación (R/P)."""

from __future__ import annotations

from solver.models import Robot

FAMILIES = ("RPRP", "RRPP", "RRRP", "RRR", "RPP", "PPP")

_DESCRIPTIONS = {
    "RPRP": "Robot de configuración RPRP: articulaciones 1 y 3 rotacionales, 2 y 4 prismáticas.",
    "RRPP": "Robot de configuración RRPP: dos rotaciones iniciales seguidas de dos prismáticas.",
    "RRRP": "Robot de configuración RRRP: tres rotaciones seguidas de una prismática.",
    "RRR": "Robot de configuración RRR: solo articulaciones rotacionales.",
    "RPP": "Robot de configuración RPP: una rotación seguida de dos prismáticas.",
    "PPP": "Robot de configuración PPP: solo articulaciones prismáticas.",
}


def validate_robot(robot: Robot) -> None:
    """Valida que el robot tenga al menos un grado de libertad y tipos R/P válidos."""
    if robot.n_dof < 1:
        raise ValueError("El robot debe tener al menos 1 grado de libertad")
    invalid = [j.kind for j in robot.joints if j.kind not in ("R", "P")]
    if invalid:
        raise ValueError(f"Tipos de articulación no válidos: {invalid}; se esperaban 'R' o 'P'")


def classify(robot: Robot) -> dict:
    """Describe la configuración del robot.

    Los índices 'rotational' y 'prismatic' son 1-based (primera articulación = 1).
    'family' es la configuración exacta si coincide con una familia conocida; en otro caso 'otro'.
    """
    validate_robot(robot)
    configuration = robot.kinds
    rotational = [i for i, j in enumerate(robot.joints, start=1) if j.kind == "R"]
    prismatic = [i for i, j in enumerate(robot.joints, start=1) if j.kind == "P"]
    family = configuration if configuration in FAMILIES else "otro"
    if family == "otro":
        descripcion = (
            f"Configuración {configuration} ({robot.n_dof} GDL, {len(rotational)} rotacionales "
            f"y {len(prismatic)} prismáticas); no corresponde a una familia conocida."
        )
    else:
        descripcion = _DESCRIPTIONS[family]
    return {
        "configuration": configuration,
        "n_dof": robot.n_dof,
        "rotational": rotational,
        "prismatic": prismatic,
        "family": family,
        "descripcion": descripcion,
    }
