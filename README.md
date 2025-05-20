## Installation

First install **PyTorch**  following the [official instructions](https://pytorch.org/get-started/locally/).

### Set up environment with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv .panda
source .panda/bin/activate
```
### Install Genesis
```bash
uv pip install 'pytransform3d[all]'
```

### Install Genesis

Clone Genesis repository and install locally:
```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
uv pip install .
cd ..
```

### Install pandaSim package

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/VahidDanesh/pandaSim.git
cd pandaSim
uv pip install .
```

## Project Structure

```
pandaSim/
├── src/pandaSim/              # Source code
│   ├── geometry/              # Geometry-related modules
│   │
│   ├── planning/              # Planning modules
│   │   └── screw_motion_planner.py # Screw motion planning
│   ├── control/               # Control modules
├── examples/                  # Example code and notebooks
│   ├── upright.ipynb          # Object reorientation example
│   └── utils.py               # Utility functions for examples
└── model/                     # Robot models and resources
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
All simulations were performed in the Genesis-AI physics-based simulator and we were able to reorient objects in multiple simulations with a xx\% success rate. 

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
[Examples/Evaluation](https://github.com/VahidDanesh/pandaSim/blob/master/examples/upright.ipynb).

<img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v3.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v5.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v4.gif" height=250px/>




