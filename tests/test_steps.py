"""Tests for the step-by-step derivations: their numbers must match the solver."""

import numpy as np
from fastapi.testclient import TestClient

from api.app import app
from solver.dh import forward_kinematics
from solver.ik import solve_planar_rrr
from solver.models import Joint, Robot
from solver.steps import fk_steps, planar_rrr_steps
from solver.steps_numeric import numeric_ik_steps

client = TestClient(app)


def planar_3r():
    return Robot("3R", tuple(Joint("R", a=1.0) for _ in range(3)))


def test_planar_steps_solutions_match_solver():
    _, steps_sols = planar_rrr_steps(1.0, 1.0, 1.0, 1.2, 0.9, 0.4)
    ref, _ = solve_planar_rrr(1.0, 1.0, 1.0, 1.2, 0.9, 0.4)
    assert len(steps_sols) == len(ref) == 2
    for a, b in zip(steps_sols, ref):
        assert np.allclose(a, b, atol=1e-9)


def test_planar_steps_unreachable_has_no_solutions():
    steps, sols = planar_rrr_steps(1.0, 1.0, 1.0, 5.0, 0.0, 0.0)
    assert sols == []
    assert "inalcanzable" in steps[-1]["lines"][0]


def test_fk_steps_final_matrix_equals_forward_kinematics():
    robot = planar_3r()
    q = [0.3, 0.5, -0.2]
    steps = fk_steps(robot, q)
    last_product = [s for s in steps if s["title"].startswith("5.3")][0]
    numbers = np.array(
        [[float(v) for v in line.strip(" []").split()] for line in last_product["lines"] if line.strip().startswith("[")]
    )
    assert np.allclose(numbers, forward_kinematics(robot, q), atol=1e-4)


def test_fk_steps_contain_every_numbered_section():
    titles = [s["title"] for s in fk_steps(planar_3r(), [0.1, 0.2, 0.3])]
    for prefix in ("1.", "2.", "3.", "4.1", "4.3", "5.1", "5.3", "6.", "7.", "8."):
        assert any(t.startswith(prefix) for t in titles), prefix


def test_numeric_steps_trace_ends_at_solution():
    robot = planar_3r()
    steps = numeric_ik_steps(robot, forward_kinematics(robot, [0.3, 0.5, -0.2]))
    titles = [s["title"] for s in steps]
    assert titles[0].startswith("1.") and titles[-1].startswith("7.")
    solution = [s for s in steps if s["title"].startswith("5.")][0]
    assert "error final de posición" in solution["lines"][1]


def test_api_fk_steps_endpoint():
    body = {"robot": {"joints": [{"kind": "R", "a": 1.0}] * 3}, "q": [0.3, 0.5, -0.2]}
    r = client.post("/api/steps/fk", json=body)
    assert r.status_code == 200
    assert len(r.json()["steps"]) > 5


def test_api_ik_steps_planar_includes_closed_form_part():
    body = {
        "robot": {"joints": [{"kind": "R", "a": 1.0}] * 3},
        "position": [1.2, 0.9, 0.0],
        "rpy_deg": [0.0, 0.0, 22.9183],
    }
    r = client.post("/api/steps/ik", json=body)
    assert r.status_code == 200
    titles = [s["title"] for s in r.json()["steps"]]
    assert titles[0].startswith("PARTE A")
    assert any(t.startswith("PARTE B") for t in titles)
