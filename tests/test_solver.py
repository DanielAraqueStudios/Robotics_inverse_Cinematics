"""Pruebas unitarias del solucionador cinemático (DH, Jacobiano, singularidades, IK, espacio de trabajo)."""

import math

import numpy as np
import pytest
import sympy as sp

from solver.classify import classify
from solver.dh import dh_matrix, forward_kinematics
from solver.ik import solve_ik, solve_planar_rrr
from solver.jacobian import analytic_jacobian, geometric_jacobian
from solver.models import Joint, Robot
from solver.report_es import build_report
from solver.singularity import analyze_position_singularity, analyze_singularity
from solver.symbolic import symbolic_chain
from solver.workspace import in_workspace_by_reach, max_reach, sample_workspace


def planar_2r(a1=1.0, a2=0.5):
    return Robot("2R", (Joint("R", a=a1), Joint("R", a=a2)))


def planar_3r(l=1.0):
    return Robot("3R", tuple(Joint("R", a=l) for _ in range(3)))


def rprp():
    return Robot(
        "RPRP",
        (
            Joint("R", alpha=math.pi / 2),
            Joint("P", alpha=-math.pi / 2),
            Joint("R", a=0.3),
            Joint("P", lower=0.0, upper=0.5),
        ),
    )


# ---------- DH y cinemática directa ----------

def test_dh_matrix_is_rotation_plus_translation():
    T = dh_matrix(0.4, 0.2, 1.1, 0.7)
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])


def test_planar_2r_fk_matches_closed_form():
    r = planar_2r(1.0, 0.5)
    q1, q2 = 0.3, 0.8
    T = forward_kinematics(r, [q1, q2])
    px = np.cos(q1) + 0.5 * np.cos(q1 + q2)
    py = np.sin(q1) + 0.5 * np.sin(q1 + q2)
    assert np.allclose(T[:2, 3], [px, py], atol=1e-12)


def test_fk_wrong_dimension_raises():
    with pytest.raises(ValueError):
        forward_kinematics(planar_2r(), [0.1])


def test_fk_rejects_non_finite():
    with pytest.raises(ValueError):
        forward_kinematics(planar_2r(), [np.nan, 0.0])


def test_prismatic_joint_translates_along_z():
    r = Robot("P", (Joint("P", lower=0.0, upper=1.0),))
    T = forward_kinematics(r, [0.7])
    assert np.allclose(T[:3, 3], [0, 0, 0.7])


# ---------- Símbolos ----------

def test_symbolic_chain_matches_numeric_fk():
    r = planar_2r(1.0, 0.5)
    q1, q2 = sp.symbols("q1 q2")
    T02 = symbolic_chain(r)[-1]
    expr_x = sp.simplify(T02[0, 3] - (sp.cos(q1) + sp.Rational(1, 2) * sp.cos(q1 + q2)))
    assert expr_x == 0


# ---------- Jacobiano ----------

def test_geometric_and_analytic_jacobian_agree():
    r = rprp()
    q = [0.4, 0.2, -0.3, 0.25]
    Jg = geometric_jacobian(r, q)
    Ja = analytic_jacobian(r, q)
    assert Jg.shape == (6, 4) == Ja.shape
    assert np.allclose(Jg, Ja, atol=1e-5)


def test_jacobian_position_rows_match_numeric_derivative():
    r = planar_3r()
    q = np.array([0.2, -0.4, 0.7])
    J = geometric_jacobian(r, q)
    h = 1e-6
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = h
        dp = (forward_kinematics(r, q + dq)[:3, 3] - forward_kinematics(r, q - dq)[:3, 3]) / (2 * h)
        assert np.allclose(J[:3, i], dp, atol=1e-6)


# ---------- Singularidades ----------

def test_planar_2r_fully_extended_is_position_singular():
    r = planar_2r()
    J = geometric_jacobian(r, [0.0, 0.0])
    res = analyze_position_singularity(J)
    assert res["status"] == "SINGULAR"
    assert res["rank"] == 1


def test_planar_2r_bent_is_not_singular():
    r = planar_2r()
    J = geometric_jacobian(r, [0.0, 0.9])
    assert analyze_position_singularity(J)["status"] == "NO_SINGULAR"
    assert analyze_singularity(J)["status"] == "NO_SINGULAR"


def test_planar_3r_bent_is_not_flagged_by_zero_z_row():
    # En un brazo plano la fila z de Jv es idénticamente nula: no debe contar como singularidad.
    r = planar_3r()
    J = geometric_jacobian(r, [0.3, 0.5, -0.2])
    assert analyze_position_singularity(J, r)["status"] == "NO_SINGULAR"


def test_planar_3r_stretched_is_position_singular():
    r = planar_3r()
    J = geometric_jacobian(r, [0.0, 0.0, 0.0])
    assert analyze_position_singularity(J, r)["status"] == "SINGULAR"


def test_analyze_singularity_reports_lost_motion_for_rank_deficient_matrix():
    J = np.array([[1.0, 2.0], [2.0, 4.0]])
    res = analyze_singularity(J)
    assert res["is_singular"] and res["rank"] == 1
    assert res["determinant"] == pytest.approx(0.0, abs=1e-12)
    assert res["null_space"].shape == (2, 1)


# ---------- Cinemática inversa ----------

def test_planar_rrr_two_solutions_reproduce_target():
    target = (1.2, 0.9, 0.4)
    sols, singular = solve_planar_rrr(1, 1, 1, *target)
    assert len(sols) == 2 and not singular
    r = planar_3r()
    for q in sols:
        T = forward_kinematics(r, q)
        assert np.allclose(T[:2, 3], target[:2], atol=1e-9)
        assert math.isclose(math.atan2(T[1, 0], T[0, 0]), target[2], abs_tol=1e-9) or math.isclose(
            math.remainder(math.atan2(T[1, 0], T[0, 0]) - target[2], 2 * math.pi), 0.0, abs_tol=1e-9
        )


def test_planar_rrr_unreachable_returns_no_solutions():
    sols, singular = solve_planar_rrr(1, 1, 1, 5.0, 0.0, 0.0)
    assert sols == [] and singular is False


def test_planar_rrr_stretched_is_flagged_singular():
    sols, singular = solve_planar_rrr(1, 1, 1, 3.0, 0.0, 0.0)
    assert singular is True
    assert len(sols) == 1


def test_numeric_ik_recovers_known_configuration():
    r = planar_3r()
    q_true = np.array([0.3, 0.5, -0.2])
    res = solve_ik(r, forward_kinematics(r, q_true))
    assert res.converged and res.reason == "ok"
    assert res.position_error < 1e-6
    target_p = forward_kinematics(r, q_true)[:3, 3]
    assert any(np.allclose(forward_kinematics(r, q)[:3, 3], target_p, atol=1e-6) for q in res.solutions)


def test_numeric_ik_unreachable_reports_reason():
    r = planar_3r()
    target = forward_kinematics(r, [0.0, 0.0, 0.0]).copy()
    target[0, 3] = 10.0
    res = solve_ik(r, target)
    assert res.converged is False
    assert res.reason == "unreachable"


def test_numeric_ik_with_prismatic_joint_converges():
    r = rprp()
    q_true = np.array([0.4, 0.3, -0.2, 0.2])
    res = solve_ik(r, forward_kinematics(r, q_true), n_random_seeds=10)
    assert res.converged
    assert res.position_error < 1e-6 and res.orientation_error < 1e-6


# ---------- Espacio de trabajo ----------

def test_max_reach_of_planar_2r():
    assert max_reach(planar_2r(1.0, 1.0)) == pytest.approx(2.0)


def test_sampled_workspace_never_exceeds_max_reach():
    r = planar_2r(1.0, 1.0)
    pts = sample_workspace(r, n_samples=2000, rng_seed=1)
    assert np.all(np.linalg.norm(pts, axis=1) <= 2.0 + 1e-9)


def test_in_workspace_by_reach():
    r = planar_2r(1.0, 1.0)
    assert in_workspace_by_reach(r, np.array([1.5, 0.0, 0.0]))
    assert not in_workspace_by_reach(r, np.array([3.0, 0.0, 0.0]))


# ---------- Clasificación ----------

def test_classify_rprp():
    res = classify(rprp())
    assert res["family"] == "RPRP"
    assert res["rotational"] == [1, 3]
    assert res["prismatic"] == [2, 4]


def test_classify_unknown_family_is_otro():
    assert classify(Robot("RPPR", (Joint("R"), Joint("P"), Joint("P"), Joint("R"))))["family"] == "otro"


def test_classify_rejects_invalid_kind():
    with pytest.raises(ValueError):
        classify(Robot("bad", (Joint("X"),)))


# ---------- Reporte ----------

def test_report_planar_2r_numbers_match_fk():
    r = planar_2r(1.0, 0.5)
    rep = build_report(r, q=[0.0, math.pi / 2])
    cd = rep["PASO_4_METODO_SOLUCION"]["cinematica_directa"]
    assert cd["px"] == pytest.approx(1.0, abs=1e-4)
    assert cd["py"] == pytest.approx(0.5, abs=1e-4)
