"""Pruebas de la API FastAPI: FK, IK y validación de entradas."""

import numpy as np
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

PLANAR_3R = {
    "name": "3R",
    "joints": [{"kind": "R", "a": 1.0}, {"kind": "R", "a": 1.0}, {"kind": "R", "a": 1.0}],
}


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_fk_planar_3r_end_position():
    r = client.post("/api/fk", json={"robot": PLANAR_3R, "q": [0.0, 0.0, 0.0]})
    assert r.status_code == 200
    body = r.json()
    assert np.allclose(body["end_position"], [3.0, 0.0, 0.0], atol=1e-9)
    assert len(body["origins"]) == 4
    assert body["singularity"] == "SINGULAR"


def test_fk_rejects_wrong_joint_count():
    r = client.post("/api/fk", json={"robot": PLANAR_3R, "q": [0.0, 0.0]})
    assert r.status_code == 422


def test_schema_rejects_inverted_limits():
    bad = {"joints": [{"kind": "P", "lower": 1.0, "upper": 0.0}]}
    r = client.post("/api/fk", json={"robot": bad, "q": [0.5]})
    assert r.status_code == 422


def test_schema_rejects_unknown_joint_kind():
    bad = {"joints": [{"kind": "X"}]}
    r = client.post("/api/fk", json={"robot": bad, "q": [0.0]})
    assert r.status_code == 422


def test_schema_rejects_too_many_joints():
    many = {"joints": [{"kind": "R", "a": 1.0}] * 13}
    r = client.post("/api/fk", json={"robot": many, "q": [0.0] * 13})
    assert r.status_code == 422


def test_ik_round_trip_through_fk_endpoint():
    fk = client.post("/api/fk", json={"robot": PLANAR_3R, "q": [0.3, 0.5, -0.2]}).json()
    pos = fk["end_position"]
    # Orientación: yaw = θ1+θ2+θ3 en grados
    yaw_deg = np.degrees(0.3 + 0.5 - 0.2)
    ik = client.post(
        "/api/ik",
        json={"robot": PLANAR_3R, "position": pos, "rpy_deg": [0.0, 0.0, yaw_deg]},
    )
    assert ik.status_code == 200
    body = ik.json()
    assert body["converged"] is True
    assert body["reason"] == "ok"
    assert len(body["solutions"]) >= 1
    for sol in body["solutions"]:
        assert np.allclose(sol["chain"]["end_position"], pos, atol=1e-5)


def test_ik_unreachable_reports_reason_not_500():
    r = client.post(
        "/api/ik",
        json={"robot": PLANAR_3R, "position": [10.0, 0.0, 0.0], "rpy_deg": [0.0, 0.0, 0.0]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["converged"] is False
    assert body["reason"] == "unreachable"
    assert body["solutions"] == []
