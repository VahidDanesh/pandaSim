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
uv sync --extra sim --extra robot --extra dev      # creates .venv; add --extra genesis for the old Genesis notebooks
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
`experiments/runs/`. `experiments/plan_box.ipynb` does the same step by step with plots.
The viewer runs on the machine that runs the script (ports 52000 and 53000).
In a pivot the hand may turn about the finger axis only as far as its palm stays clear of the box: the
limits of the bearing angle phi follow from the grasp geometry (`planning/clearance.py`) and are enforced by
the QP.

## Project Structure

```
pandaSim/
├── src/pandaSim/                   # Source code
│   ├── geometry/                   # Geometry-related modules
│   │   └── genesis_adapter.py      # Genesis adapter
│   ├── planning/                   # Planning modules
│   │   └── screw_motion_planner.py # Screw motion planning
│   ├── control/                    # Control modules
|   |   └── resolved_rate.py        # RMRC motion controller
├── examples/                       # Example code and notebooks
│   ├── upright.ipynb               # Object reorientation example
│   └── utils.py                    # Utility functions for examples
└── model/                          # Robot models and resources
```

## Core Components

### Genesis Geometry Adapter

The `GenesisAdapter` module provides an interface to the Genesis physics engine's geometry representation. It allows extracting geometric features like size, edges, and vertices from objects in the simulation, which are essential for motion planning. There will be `Robotic Toolbox for Python` adapter in future.

### Screw Motion Planner

The `ScrewMotionPlanner` implements the core manipulation planning functionality. It computes screw axes for object manipulation and generates trajectories that allow the robot to reorient objects. The planner supports rotational slippage at the grasp point to maximize manipulability during object reorientation.

## pandaSim Project

We utilized a Tabletop Franka Panda robot to reorient various fallen objects, including cuboids, cylinders, and a mustard bottle. 
We assumed that the objects' dimensions and initial configurations lie within the robot's reachable workspace, and that the robot is capable of performing the necessary manipulations, including solving the inverse kinematics, from those initial poses. 
In each scene, the initial configurations of the objects were randomly assigned. Using the functions available in the Genesis Geometry Adapter, the geometric features—such as size, edges and vertices—were extracted, from which all necessary parameters for motion execution were derived.
This allows to construct the screw axis which the object will be manipulated about that axis.
Moreover, an extension involving rotational slippage at the grasp point to allow upright manipulation while maximizing manipulability was explored.
All simulations were performed in the Genesis-AI physics-based simulator and we were able to reorient objects in multiple simulations with a 96\% success rate. 

## Examples

The project includes a comprehensive example notebook that demonstrates the full manipulation pipeline:

- [upright.ipynb](https://github.com/VahidDanesh/pandaSim/blob/master/examples/upright.ipynb): Shows how to use the system to reorient various objects (cylinders, cubes, bottles) using the Franka Panda robot.

The example demonstrates:
1. Initialize the scene and robot
2. Identify and grasp the object
3. Plan and execute screw motion trajectories
4. Release the object in its upright position
5. Repeat for multiple objects

## Gallery

The following visualize reorientation of objects with help of environment and rotational slippage at finger level in two senarios.
The code for these examples can be found in
[Examples/upright](https://github.com/VahidDanesh/pandaSim/blob/master/examples/upright.ipynb).

<img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v3.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v5.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v4.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v1.gif" height=250px/>




