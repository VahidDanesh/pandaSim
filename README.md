## Installation

First install **PyTorch**  following the [official instructions](https://pytorch.org/get-started/locally/).


Use pip to install the pytranform3d package from PyPI:

```bash
pip install 'pytransform3d[all]'
```

Clone Genesis repository and install locally:
```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install .
```

install pandaSim package:
```bash
```

## pandaSim Project

we utilized a Tabletop Franka Panda robot to reorient various fallen objects, including cuboids, cylinders, and a mustard bottle. 
We assumed that the objects' dimensions and initial configurations lie within the robot’s reachable workspace, and that the robot is capable of performing the necessary manipulations, including solving the inverse kinematics, from those initial poses. 
In each scene, the initial configurations of the objects were randomly assigned. Using the functions available in [reference/module name], the geometric features—such as size, edges and vertices—were extracted, from which all necessary parameters for motion execution were derived.
This allows to construct the screw axis which the object will be manipulated about that axis.
Moreover, an extension involving rotational slippage at the grasp point to allow upright manipulation while maximizing manipulability was explored.
All simulations were performed in the Genesis-AI physics-based simulator and we were able to reorient objects in multiple simulations with a xx\% success rate. 


## Gallery

The following visualize reorientation of objects with help of environment and rotational slippage at finger level in two senarios.
The code for these examples can be found in
[Examples/Evaluation](https://github.com/VahidDanesh/pandaSim/blob/master/examples/upright.ipynb).

<img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v3.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v5.gif" height=250px/>            <img src="https://github.com/VahidDanesh/pandaSim/blob/master/examples/videos/v4.gif" height=250px/>




