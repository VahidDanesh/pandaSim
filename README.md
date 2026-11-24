## Installation

The repository is self-contained: `uv.lock` pins every package and `vendor/` holds the
panda-py build that matches the lab robot (libfranka 0.8.0; the PyPI wheel bundles 0.9.2).
Linux x86-64, Python 3.10 (uv downloads it if missing).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/VahidDanesh/pandaSim.git
cd pandaSim
unset PYTHONPATH                                   # a sourced ROS shadows the venv's numpy
uv sync --extra sim --extra robot --extra dev      # creates .venv
uv run pytest
```

- `sim`: Robotics Toolbox, Swift, the QP solver (kinematic simulation and planning).
- `robot`: panda-py for the real Franka. Copy `.env.example` to `.env` and fill in the Desk credentials.
- `dev`: pytest, Jupyter kernel.

Run scripts with `uv run python ...` or `.venv/bin/python ...`.

## Box planner in simulation

Plan a sequence of pivots and slides that brings a box from a start pose to a goal pose, execute it
on the simulated arm and check the result. Tasks and parameters are in
`experiments/configs/box_plan.yaml`.

```bash
.venv/bin/python experiments/run_plan.py --task all                 # all tasks, no viewer
.venv/bin/python experiments/run_plan.py --task slide_pivot --viewer  # replay in Swift (browser tab)
.venv/bin/python experiments/run_plan.py --task all --rigid         # rigid grasp instead of free rotation
.venv/bin/python experiments/run_plan.py --task all --rigid --plan-free-rotation  # rigid execution of free-rotation plans
```

`--free-rotation / --rigid` sets the execution model of the pivots, `--plan-free-rotation / --plan-rigid`
the model of the planner's feasibility check (default: the same as execution). The logs go to
`experiments/runs/`. `experiments/plan_box.ipynb` is a tour of the whole project in simulation (primitives, contact QP,
sequencing, gaiting, free vs rigid, pre-flight, the protocol of the paper, a playground for your own task), with
Swift inside the notebook cells.
The viewer runs on the machine that runs the script (ports 52000 and 53000).
In a pivot the hand may turn about the finger axis only as far as its palm stays clear of the box: the
limits of the bearing angle phi follow from the grasp geometry (`planning/clearance.py`) and are enforced by
the QP.

## From a start pose to the real robot

Step-by-step procedure for the robot: [docs/HARDWARE_DAY.md](docs/HARDWARE_DAY.md). Overview of the code and the sequencer: [docs/GUIDE.md](docs/GUIDE.md).

`experiments/run_box.py` takes the box's start and goal pose (`x,y,yaw_deg,face`; face is `lying`, `lying_flip`, `standing` or `standing_flip`), plans, simulates the plan with the executor that runs on the robot, and prints a pre-flight report (GO / NO-GO):

```bash
python experiments/run_box.py --start 0.50,0.00,0,lying --goal 0.45,-0.15,60,standing            # no robot
python experiments/run_box.py --start 0.50,0.00,0,lying --goal 0.45,-0.15,60,standing --replay   # Swift
python experiments/run_box.py --start ... --goal ... --execute --host 172.16.0.2 --level air       # robot
```

The run folder `experiments/runs/box/<name>/` gets `plan.json`, `expected.npz` (the simulated run, for the Swift replay), `preflight.json` and, after a robot run, `real.npz` and `compare.json` (joint tracking against the simulation, forces, loop timing, object pose by forward kinematics at the release). Edit `experiments/configs/box_real.yaml` first: weigh the box (mass), check the size and the robot base height.

`--gait` plans object gaiting (the box turns about its vertices on the same face) instead of slides and pivots; `--rigid` executes the free-rotation plan with the rigid grasp. The experiments of the paper are listed in `experiments/configs/protocol.yaml`; `experiments/protocol.py` checks all of them in simulation and prints their commands.

`--execute` connects to the robot, checks it (joint limits, errors, load, distance to the ready configuration, gripper), prints the plan and moves only after you type `RUN <level> <run name>`; there is no flag that skips this. Levels: `air` (no box, fingers close to the box width in the air), `low` and `full` (box on the table); `--speed` scales the approach, slides and pivots (0.3 by default for `air` and `low`). A robot error, a wrench above the limits, an infeasible QP or a failed grasp stops the arm and leaves the gripper as it is.

## Project Structure

```
pandaSim/
├── src/pandaSim/
│   ├── planning/        se3.py, box_primitives.py (pivot and slide plans), sequencer.py (sequences of slides and
│   │                    pivots), gait.py (object gaiting with one arm), feasibility.py (rollout of a step with the
│   │                    executor), clearance.py (palm and phi limits), collision.py (arm vs box and table), primitives.py
│   ├── control/         contact_qp.py (the QP of the paper), primitive_executor.py (run_plan: segments, gripper,
│   │                    safety stops), backends.py (SimBackend, PandaBackend), viewer.py (Swift live view and replay)
│   └── geometry/        utils.py (Panda model with the virtual bearing joint)
├── experiments/         run_box.py (start/goal -> plan -> pre-flight -> robot), run_plan.py, protocol.py, configs/,
│                        analysis and paper tools, plan_box.ipynb (tour of the project in simulation)
├── tests/unit/          pytest
├── docs/                GUIDE.md, HARDWARE_DAY.md, HARDWARE_SAFETY.md
├── model/               Panda URDF with the virtual bearing joint, meshes
└── vendor/              panda-py built against libfranka 0.8.0
```

## Earlier work

The first version of this repository reoriented fallen objects (cuboids, cylinders, a mustard bottle) with screw
motions and rotational slippage at the fingertips in the Genesis simulator (96% success rate over random scenes). That
code, the `upright.ipynb` example and its videos are on the
[`master` branch](https://github.com/VahidDanesh/pandaSim/tree/master); they were removed from this branch, which
holds the box planner and the hardware path.

<img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v3.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v5.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v4.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v1.gif" height=250px/>
