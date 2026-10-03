# Robotics Inverse Kinematics

A kinematics toolkit for serial robot chains and biomechatronic limb models. It computes forward kinematics from Denavit–Hartenberg (DH) parameters, solves inverse kinematics numerically, analyses singularities and workspace, and presents the results in a desktop interface backed by a local HTTP API.

## Features

- **Forward kinematics** from standard DH parameters, with revolute (`R`) and prismatic (`P`) joints and per-joint limits.
- **Inverse kinematics**
  - Closed-form solver for the planar 3R arm, returning both elbow-up and elbow-down solutions.
  - Bounded numerical solver for general chains, using multiple seeds, joint limits and explicit convergence reasons.
- **Jacobians**: geometric and analytic (numerical differentiation), plus symbolic forms with SymPy.
- **Singularity analysis** by singular value decomposition. It reports the rank, the null space, the lost-motion directions and a plain-language message.
- **Workspace**: maximum reach, sampled workspace and a reach-based membership test.
- **Classification** of robot configurations (`RPRP`, `RRPP`, `RRRP`, `RRR`, `RPP`, `PPP`).
- **Step-by-step solutions** with every number substituted: DH table, each link and cumulative transform, the end-effector pose, validity and singularity checks. For the planar 3R arm the closed-form derivation is included; for any chain the numerical IK iteration trace is shown.
- **Symbolic DH tables and step-by-step matrix products** (SymPy), with a Spanish-language report builder.
- **REST API** (FastAPI) for forward and inverse kinematics.
- **Desktop interface** (PyQt6) with forward and inverse modes, a 3D chain viewer with animated transitions, and threaded requests so the window never blocks.

## Project structure

```
Robotics_inverse_Cinematics/
├── requirements.txt
├── README.md
├── src/
│   ├── main.py              # Entry point: starts the API and opens the interface
│   ├── solver/              # Kinematics core (no web or GUI dependencies)
│   │   ├── models.py        # Robot and Joint data structures
│   │   ├── dh.py            # DH matrices and forward kinematics
│   │   ├── jacobian.py      # Geometric, analytic and symbolic Jacobians
│   │   ├── singularity.py   # SVD-based singularity analysis
│   │   ├── ik.py            # Planar 3R closed form and bounded numerical IK
│   │   ├── workspace.py     # Reach, sampling and membership checks
│   │   ├── classify.py      # Configuration family classification
│   │   ├── symbolic.py      # Symbolic DH tables and chain products
│   │   └── report_es.py     # Spanish step-by-step report builder
│   ├── api/                 # FastAPI application
│   │   ├── app.py           # Endpoints and response assembly
│   │   └── schemas.py       # Request and response validation
│   └── ui/                  # PyQt6 interface
│       ├── main_window.py   # Window, FK/IK modes and feedback
│       ├── viewer.py        # Matplotlib 3D chain viewer
│       ├── api_client.py    # Non-blocking HTTP client (thread pool)
│       ├── presets.py       # Example robots
│       └── style.qss        # Dark engineering theme
└── tests/
    ├── conftest.py
    ├── test_solver.py       # Solver unit tests
    └── test_api.py          # API tests
```

## Requirements

- Python 3.12 or newer (tested with 3.12 and 3.14)
- Windows, macOS or Linux with a display for the desktop interface

## Setup

1. **Clone the repository and open its folder**

   ```bash
   cd Robotics_inverse_Cinematics
   ```

2. **Create a virtual environment** (recommended, keeps the packages isolated):

   ```bash
   python -m venv .venv
   ```

   Activate it:

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Windows (cmd): `.venv\Scripts\activate.bat`
   - macOS / Linux: `source .venv/bin/activate`

3. **Install the dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Check the installation** by running the test suite:

   ```bash
   python -m pytest tests/ -q
   ```

   All tests should pass.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'uvicorn'`** (or `PyQt6`, `fastapi`, …): the interpreter you are running doesn't have the dependencies. Install them with that same interpreter, for example:

  ```bash
  C:\path\to\python.exe -m pip install -r requirements.txt
  ```

  Then run the app with that same `python.exe`.
- **Several Python versions installed**: use `python --version` and `where python` (Windows) or `which -a python3` (macOS / Linux) to see which one is active.
- **`pip` warns that a script is not on PATH**: this is harmless when you call Python with `python -m ...` or with its full path.
- **The window opens but shows a connection error**: the API did not start. Close the window, make sure no other program is using the port, and run `python main.py` again.

## Running

Start the application from the `src` folder. This launches the API on a free local port and then opens the interface:

```bash
cd src
python main.py
```

To run the API on its own (for example, to call it from another program):

```bash
cd src
python -m uvicorn api.app:app --port 8000
```

Interactive API documentation is then available at `http://127.0.0.1:8000/docs`.

## Using the interface

- **Robot**: choose a preset in the header. Presets are defined in `src/ui/presets.py`.
- **Forward kinematics (FK)**: move the joint sliders or type values in the spinboxes. Rotational joints are shown in degrees and prismatic joints in meters. The pose updates as you change values.
- **Inverse kinematics (IK)**: enter the target position (x, y, z in meters) and orientation (roll, pitch, yaw in degrees), then press *Calcular cinemática inversa*. Each solution appears in the list. Selecting one applies it to the joints and the viewer.
- **Feedback**: the status bar reports the outcome in plain language. A warning appears when the configuration is singular. Unreachable targets, non-convergence and connection problems are reported with their causes.
- **Viewer**: drag with the mouse to rotate the 3D view.
- **Step-by-step solution**: press *Show step-by-step solution* to open a window with the full derivation of the current forward or inverse problem, with the numbers substituted and a *Copy all* button.

## API reference

All endpoints accept and return JSON.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `POST` | `/api/fk` | Forward kinematics for given joint values |
| `POST` | `/api/ik` | Inverse kinematics for a target pose |
| `POST` | `/api/steps/fk` | Step-by-step forward-kinematics derivation |
| `POST` | `/api/steps/ik` | Step-by-step inverse-kinematics derivation (closed form for planar 3R, plus numerical trace) |

### Robot description

```json
{
  "name": "3R",
  "joints": [
    {"kind": "R", "a": 1.0},
    {"kind": "R", "a": 1.0},
    {"kind": "R", "a": 1.0}
  ]
}
```

Fields per joint: `kind` (`"R"` or `"P"`, required), `a`, `alpha`, `theta`, `d` (DH constants, default 0), and optional `lower` and `upper` limits. Up to 12 joints are accepted. Angles are in radians and lengths in meters.

### Forward kinematics

`POST /api/fk`

```json
{"robot": {...}, "q": [0.3, 0.5, -0.2]}
```

Response: `origins` (base and each frame origin), `end_position`, `rotation`, `matrix` (4×4 homogeneous transform), `singularity` (`"SINGULAR"` or `"NO_SINGULAR"`) and `singularity_message`.

### Inverse kinematics

`POST /api/ik`

```json
{
  "robot": {...},
  "position": [2.477, 1.578, 0.0],
  "rpy_deg": [0.0, 0.0, 34.38],
  "seed": [0.3, 0.5, -0.2]
}
```

`seed` is optional. Response: `converged`, `reason` (`ok`, `unreachable`, `max_iter`, `out_of_limits` or `singular`), `singular_flag`, `position_error`, `orientation_error` (`null` when not finite) and `solutions`, each with its joint values and full chain state. The solve is limited to 5 seconds; longer requests return HTTP 504.

## Using the solver from Python

```python
import numpy as np
from solver.models import Robot, Joint
from solver.dh import forward_kinematics
from solver.ik import solve_ik
from solver.jacobian import geometric_jacobian
from solver.singularity import analyze_position_singularity

arm = Robot("3R", tuple(Joint("R", a=1.0) for _ in range(3)))

T = forward_kinematics(arm, [0.3, 0.5, -0.2])
result = solve_ik(arm, T)                      # reproduces the same pose
print(result.reason, result.solutions[0])

J = geometric_jacobian(arm, [0.0, 0.0, 0.0])
print(analyze_position_singularity(J, arm)["status"])   # SINGULAR (fully stretched)
```

The examples assume `src` is on the import path: run them from the `src` folder, or add it with `sys.path.insert(0, "src")`. The test suite sets the path itself.

## Testing

```bash
python -m pytest tests/ -v
```

The suite covers DH transforms, forward kinematics against closed forms, Jacobian consistency with numerical derivatives, singularity detection, closed-form and numerical inverse kinematics (including round trips and unreachable targets), workspace bounds, configuration classification, the report builder and the API endpoints, including input validation.

## Conventions

- **DH convention**: standard (Craig/Spong). Each link is `Rz(θ)·Tz(d)·Tx(a)·Rx(α)`, so `T_{i-1,i}` uses the joint's DH parameters. The joint variable adds to `θ` for revolute joints and to `d` for prismatic joints.
- **Orientation**: roll, pitch and yaw are ZYX Euler angles in degrees.
- **Units**: radians and meters internally; the interface displays degrees for rotational joints.

## Limitations

- **Singularity threshold**: singularities are judged against the rank the arm reaches in general configurations, with a relative tolerance of 1e-4. A configuration that is singular only within that tolerance is still flagged.
- **Numerical IK**: general chains are solved by bounded least squares from several seeds. It finds a solution when one exists within the iteration budget, but it doesn't prove that no solution exists when it fails. Closed-form solutions are provided for the planar 3R arm only.
- **Joint limits** are enforced during IK and in the interface. The forward kinematics endpoint doesn't reject out-of-limit inputs.
- **Workspace sampling** is an estimate. The reach test is necessary but not sufficient for membership; exact membership requires an IK solve.
- **Dynamics and friction** are not modelled. Links are rigid.
- **Numerical IK steps** show the least-squares trace (one row per residual evaluation, including finite-difference evaluations). They describe what the solver did, not a closed-form derivation.
- **Step-by-step output is in English.** The Spanish report builder (`solver/report_es.py`) is separate and not yet connected to the window.

## License

No license has been chosen yet. All rights are reserved by the author until one is added.
