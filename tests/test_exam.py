"""Pruebas de la convención DH modificada, marcos, Jacobiano y endpoint de examen."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from solver.dh import forward_kinematics  # noqa: E402
from solver.exam import analyze_exam  # noqa: E402
from solver.mdh import forward_kinematics_mdh, mdh_to_standard_robot  # noqa: E402
from solver.models import Joint, Robot  # noqa: E402


def planar_2r_modified(l1: float, l2: float) -> tuple[Robot, tuple[float, float]]:
    """2R plano en DH modificada. El eslabón L2 se pasa como herramienta (alpha, a)."""
    robot = Robot("2R", (Joint("R"), Joint("R", a=l1, alpha=0.0)))
    return robot, (0.0, l2)


def test_planar_2r_modified_matches_closed_form():
    l1, l2 = 0.7, 0.4
    robot, tool = planar_2r_modified(l1, l2)
    rng = np.random.default_rng(1)
    for _ in range(20):
        q1, q2 = rng.uniform(-math.pi, math.pi, 2)
        T = forward_kinematics_mdh(robot, [q1, q2], tool=tool)
        px = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
        py = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
        assert T[0, 3] == pytest.approx(px, abs=1e-12)
        assert T[1, 3] == pytest.approx(py, abs=1e-12)
        assert T[2, 3] == pytest.approx(0.0, abs=1e-12)
        assert T[:3, 2] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


def test_modified_and_standard_give_same_end_pose_for_3r_planar():
    """Mapeo: en DH modificada, (alpha_{j-1}, a_{j-1}) de la articulación j son los
    (alpha_{j-1}, a_{j-1}) estándar del eslabón j-1, con alpha_0 = a_0 = 0.
    El último eslabón estándar (a3) queda como marco de herramienta fijo.
    """
    a1, a2, a3 = 0.5, 0.4, 0.3
    std = Robot("3R", (
        Joint("R", a=a1), Joint("R", a=a2), Joint("R", a=a3),
    ))
    mod = Robot("3R-mod", (
        Joint("R", a=0.0, alpha=0.0),
        Joint("R", a=a1, alpha=0.0),
        Joint("R", a=a2, alpha=0.0),
    ))
    tool = (0.0, a3)
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = rng.uniform(-math.pi, math.pi, 3)
        assert forward_kinematics_mdh(mod, q, tool=tool) == pytest.approx(forward_kinematics(std, q), abs=1e-12)


def test_modified_and_standard_equal_for_general_spatial_mapping():
    """Mapeo general con alpha, a, d y theta no nulos; conversión inversa exacta.

    Estándar:  J1 R(a=.2, alpha=.3, d=.1)  J2 P(a=.5, alpha=-.7, theta=.4)  J3 R(a=.25, alpha=.9, d=-.2)
    Modificada: theta/d de la articulación j = los del estándar j;
                alpha_{j-1}, a_{j-1} = los del estándar j-1; alpha_0 = a_0 = 0; herramienta = (alpha_3, a_3).
    """
    std = Robot("gen", (
        Joint("R", a=0.2, alpha=0.3, d=0.1),
        Joint("P", a=0.5, alpha=-0.7, theta=0.4),
        Joint("R", a=0.25, alpha=0.9, d=-0.2),
    ))
    mod = Robot("gen-mod", (
        Joint("R", a=0.0, alpha=0.0, d=0.1),
        Joint("P", a=0.2, alpha=0.3, theta=0.4),
        Joint("R", a=0.5, alpha=-0.7, d=-0.2),
    ))
    tool = (0.9, 0.25)
    rng = np.random.default_rng(3)
    for _ in range(10):
        q = rng.uniform(-1.0, 1.0, 3)
        assert forward_kinematics_mdh(mod, q, tool=tool) == pytest.approx(forward_kinematics(std, q), abs=1e-12)
    back = mdh_to_standard_robot(mod, tool=tool)
    for j_back, j_std in zip(back.joints, std.joints):
        assert j_back.kind == j_std.kind
        assert (j_back.a, j_back.alpha, j_back.theta, j_back.d) == pytest.approx(
            (j_std.a, j_std.alpha, j_std.theta, j_std.d)
        )


def rprp_robot() -> Robot:
    return Robot("RPRP", (
        Joint("R", a=0.0, alpha=0.0),
        Joint("P", a=0.0, alpha=math.pi / 2, theta=0.0),
        Joint("R", a=0.3, alpha=-math.pi / 2),
        Joint("P", a=0.0, alpha=math.pi / 2),
    ))


def test_jacobian_jv_matches_numerical_derivative_rprp():
    robot = rprp_robot()
    q = np.array([0.4, 0.25, -0.6, 0.15])
    h = 1e-6
    res = analyze_exam(robot, q, "modified")
    Jv = np.array(res["jacobiano"]["Jv"])
    for i in range(robot.n_dof):
        dq = np.zeros(4)
        dq[i] = h
        p_plus = forward_kinematics_mdh(robot, q + dq)[:3, 3]
        p_minus = forward_kinematics_mdh(robot, q - dq)[:3, 3]
        numeric = (p_plus - p_minus) / (2 * h)
        assert Jv[:, i] == pytest.approx(numeric, abs=1e-6)


def test_exam_contains_spanish_keys_and_frames():
    res = analyze_exam(rprp_robot(), [0.1, 0.2, 0.3, 0.4], "modified")
    for key in ("clasificacion", "grados_libertad", "articulaciones", "marcos", "tabla_dh",
                "matrices", "fk", "jacobiano", "singularidad", "verificacion"):
        assert key in res
    assert res["clasificacion"]["family"] == "RPRP"
    assert res["articulaciones"] == {"rotacionales": [1, 3], "prismaticas": [2, 4]}
    assert len(res["marcos"]) == 5
    assert res["marcos"][0]["name"] == "{0}"
    assert "descripcion" in res["marcos"][1]
    assert res["verificacion"]["dimensiones_ok"] is True
    assert res["verificacion"]["ortogonalidad"] < 1e-10
    assert res["verificacion"]["det_R"] == pytest.approx(1.0)


def test_endpoint_analyze_returns_200_and_spanish_keys():
    robot = rprp_robot()
    payload = {
        "robot": {"name": "RPRP", "joints": [
            {"kind": j.kind, "a": j.a, "alpha": j.alpha, "theta": j.theta, "d": j.d}
            for j in robot.joints
        ]},
        "q": [0.1, 0.2, 0.3, 0.4],
    }
    client = TestClient(app)
    resp = client.post("/api/exam/analyze", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["convencion"] == "modified"
    for key in ("clasificacion", "marcos", "tabla_dh", "matrices", "fk", "jacobiano", "singularidad", "verificacion"):
        assert key in body
    assert "mensaje_es" in body["singularidad"]


def test_endpoint_standard_convention_and_validation_422():
    robot = rprp_robot()
    joints = [{"kind": j.kind, "a": j.a, "alpha": j.alpha, "theta": j.theta, "d": j.d} for j in robot.joints]
    client = TestClient(app)
    ok = client.post("/api/exam/analyze", json={"robot": {"joints": joints}, "q": [0, 0, 0, 0], "convention": "standard"})
    assert ok.status_code == 200
    assert ok.json()["convencion"] == "standard"
    bad_conv = client.post("/api/exam/analyze", json={"robot": {"joints": joints}, "q": [0, 0, 0, 0], "convention": "dh"})
    assert bad_conv.status_code == 422
    bad_len = client.post("/api/exam/analyze", json={"robot": {"joints": joints}, "q": [0, 0]})
    assert bad_len.status_code == 422
