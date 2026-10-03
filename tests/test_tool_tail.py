"""Pruebas del marco de herramienta fijo (tool tail) tras la última articulación."""

import math

import numpy as np
import pytest

from fastapi.testclient import TestClient

from api.app import app
from solver.dh import forward_kinematics
from solver.exam import analyze_exam
from solver.ik import solve_ik
from solver.jacobian import geometric_jacobian
from solver.mdh import forward_kinematics_mdh, mdh_to_standard_robot
from solver.models import Joint, Robot
from solver.workspace import max_reach

client = TestClient(app)


def tailed_2r_standard() -> Robot:
    """2R plano con a=(1,1) y herramienta a=0.5 a lo largo de x (convención estándar)."""
    return Robot("2R-tool", (Joint("R", a=1.0), Joint("R", a=1.0)), tool=(0.0, 0.0, 0.5, 0.0))


def tailed_2r_modified() -> Robot:
    """Mismo brazo en DH modificada: a_1 = 1 entre articulaciones y herramienta a = 1.5."""
    return Robot("2R-tool-mod", (Joint("R"), Joint("R", a=1.0)), tool=(0.0, 0.0, 1.5, 0.0))


def tailed_2r_modified_general() -> Robot:
    """2R modificado con herramienta general (theta, d, a, alpha) para la identidad de conversión."""
    return Robot(
        "2R-general",
        (Joint("R"), Joint("R", a=0.6, alpha=0.2, theta=0.1, d=0.05)),
        tool=(0.3, 0.2, 0.5, 0.4),
    )


def test_planar_2r_with_tool_reaches_2_5_standard_and_modified():
    std = tailed_2r_standard()
    p_std = forward_kinematics(std, [0.0, 0.0])[:3, 3]
    assert p_std == pytest.approx([2.5, 0.0, 0.0], abs=1e-12)

    mod = tailed_2r_modified()
    p_mod = forward_kinematics_mdh(mod, [0.0, 0.0])[:3, 3]
    assert p_mod == pytest.approx([2.5, 0.0, 0.0], abs=1e-12)


def test_tool_default_is_none_and_has_tool_flag():
    plain = Robot("2R", (Joint("R", a=1.0), Joint("R", a=1.0)))
    assert plain.tool == (0.0, 0.0, 0.0, 0.0)
    assert not plain.has_tool
    assert tailed_2r_standard().has_tool


def test_geometric_jacobian_matches_central_differences_of_fk():
    robot = tailed_2r_standard()
    rng = np.random.default_rng(3)
    q = rng.uniform(-math.pi, math.pi, 2)
    J = geometric_jacobian(robot, q)
    assert J.shape == (6, 2)
    h = 1e-6
    for i in range(2):
        dq = np.zeros(2)
        dq[i] = h
        dp = (forward_kinematics(robot, q + dq)[:3, 3] - forward_kinematics(robot, q - dq)[:3, 3]) / (2 * h)
        assert J[:3, i] == pytest.approx(dp, abs=1e-6)


def test_mdh_to_standard_identity_with_tail():
    robot = tailed_2r_modified_general()
    std = mdh_to_standard_robot(robot)
    assert std.tool == (0.3, 0.2, 0.0, 0.0)
    rng = np.random.default_rng(7)
    for _ in range(10):
        q = rng.uniform(-math.pi, math.pi, 2)
        T_mod = forward_kinematics_mdh(robot, q)
        T_std = forward_kinematics(std, q)
        assert np.max(np.abs(T_mod - T_std)) < 1e-12


def test_ik_round_trip_reproduces_tailed_position():
    robot = tailed_2r_standard()
    q_true = np.array([0.4, -0.9])
    target = forward_kinematics(robot, q_true)
    result = solve_ik(robot, target)
    assert result.converged
    p_sol = forward_kinematics(robot, result.solutions[0])[:3, 3]
    assert np.linalg.norm(p_sol - target[:3, 3]) < 1e-6


def test_max_reach_includes_tool_offsets():
    assert max_reach(tailed_2r_standard()) == pytest.approx(2.5)


def test_api_fk_returns_end_position_with_tool():
    body = {
        "robot": {"joints": [{"kind": "R", "a": 1.0}, {"kind": "R", "a": 1.0}], "tool": [0.0, 0.0, 0.5, 0.0]},
        "q": [0.0, 0.0],
    }
    resp = client.post("/api/fk", json=body)
    assert resp.status_code == 200
    assert resp.json()["end_position"] == pytest.approx([2.5, 0.0, 0.0], abs=1e-12)


def test_api_rejects_tool_with_wrong_length():
    body = {
        "robot": {"joints": [{"kind": "R", "a": 1.0}], "tool": [0.0, 0.0, 0.5]},
        "q": [0.0],
    }
    assert client.post("/api/fk", json=body).status_code == 422


def test_exam_analyze_with_tool_counts_frames_and_matrices():
    robot = tailed_2r_standard()
    n = robot.n_dof
    data = analyze_exam(robot, [0.1, 0.2], "standard")
    # n+1 eslabones (incluye la herramienta) + n+1 acumulados (T01..T0,n+1)
    assert len(data["matrices"]) == 2 * (n + 1)
    assert len(data["marcos"]) == n + 2
    tail = data["marcos"][-1]
    assert tail["name"] == "{3}"
    assert tail["joint_axis"] is None
    assert tail["descripcion"] == "Marco de herramienta (fijo)"
    labels = [m["label"] for m in data["matrices"]]
    assert "T23" in labels and "T03" in labels
    assert len(data["jacobiano"]["columnas"]) == n
    assert data["fk"]["p"] == pytest.approx(
        (forward_kinematics(robot, [0.1, 0.2])[:3, 3]).tolist(), abs=1e-12
    )


def test_exam_analyze_endpoint_with_tool_returns_200():
    body = {
        "robot": {"joints": [{"kind": "R", "a": 1.0}, {"kind": "R", "a": 1.0}], "tool": [0.0, 0.0, 0.5, 0.0]},
        "q": [0.0, 0.0],
        "convention": "standard",
    }
    resp = client.post("/api/exam/analyze", json=body)
    assert resp.status_code == 200
    data = resp.json()
    # 2 joints + 1 tool link = 3 links, plus 3 cumulative frames = 6 matrices
    assert len(data["matrices"]) == 6
    assert data["fk"]["p"] == pytest.approx([2.5, 0.0, 0.0], abs=1e-12)


def test_exam_analyze_endpoint_modified_with_tool_returns_200():
    body = {
        "robot": {"joints": [{"kind": "R"}, {"kind": "R", "a": 1.0}], "tool": [0.0, 0.0, 1.5, 0.0]},
        "q": [0.0, 0.0],
    }
    resp = client.post("/api/exam/analyze", json=body)
    assert resp.status_code == 200
    assert resp.json()["fk"]["p"] == pytest.approx([2.5, 0.0, 0.0], abs=1e-12)
