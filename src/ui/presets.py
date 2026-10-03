"""Robots de ejemplo que la interfaz envía a la API (mismo formato que RobotIn)."""

PRESETS = {
    "Brazo plano 3R (1, 1, 1 m)": {
        "name": "3R",
        "joints": [
            {"kind": "R", "a": 1.0},
            {"kind": "R", "a": 1.0},
            {"kind": "R", "a": 1.0},
        ],
    },
    "Brazo RPRP con prismática": {
        "name": "RPRP",
        "joints": [
            {"kind": "R", "alpha": 1.5707963267948966},
            {"kind": "P", "alpha": -1.5707963267948966, "lower": 0.0, "upper": 0.5},
            {"kind": "R", "a": 0.3},
            {"kind": "P", "lower": 0.0, "upper": 0.5},
        ],
    },
}
