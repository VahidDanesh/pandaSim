
# Problem statement

In general, any rigid body displacement can be expressed as a rotation $\theta \in \mathbb{R}$ about a screw axis $\mathcal{S}$ followed by a translation $d \in \mathbb{R}$ along the axis. By representing the screw axis $\mathcal{S}$ by a unit vector $\boldsymbol{u} \in \mathbb{R}^3$ along the axis and an arbitrary point $\boldsymbol{r} \in \mathbb{R}^3$ on the axis, the screw parameters, using Plücker coordinates, are defined as $(\theta,d,\boldsymbol{u},\boldsymbol{m})$, where $\boldsymbol{m}= \boldsymbol{r} \times \boldsymbol{u} \in \mathbb{R}^3$. Therefore, the screw motions can be efficiently expressed by the dual quaternions as $ D_T = Q_R + \epsilon \frac{1}{2} Q_P Q_R =( \cos\frac{\theta}{2} + \boldsymbol{u}\sin\frac{\theta}{2}) + \epsilon\left(-\frac{d}{2}\sin\frac{\theta}{2}+\sin\frac{\theta}{2}\boldsymbol{m}+\frac{d}{2}\cos\frac{\theta}{2}\boldsymbol{u}\right)$. In our case which is a \textit{constant} screw motion, $\boldsymbol{u}$ and $\boldsymbol{m}$ remains constant and only $\theta$ changes.  The smooth path in $SE(3)$ provided by the ScLERP is derived by $D(s) = D_1D_{12}^{s}$, where $s \in [0, 1]$ is a scalar path parameter and $D_{12}$ is the transformation of $\mathcal{C}_2$ with respect to $\mathcal{C}_1$. $D_{12}^ {s}$ can be computed using $D_{12}^{s} = \left(\cos \frac{s \theta}{2}, \sin \frac{s\theta}{2}\boldsymbol{u}\right) + \epsilon \left(- \frac{s d}{2} \sin \frac{s \theta}{2}, \frac{s d}{2}\cos \frac{s \theta}{2} \boldsymbol{u} + \sin \frac{s \theta}{2} \boldsymbol{m}\right)$ after extracting the screw parameters $\boldsymbol{u}$, $\boldsymbol{m}$, $\theta$, $d$ from $D_{12}$.

If $\mathcal{C}_1$ and $\mathcal{C}_2$ have a contact edge in common, the final pose can be achieved by pivoting the object about the common edge.

We will also explore an advanced method in which the manipulator induces object motion through rotational slippage at the grasp point while maintaining a fixed (or nearly fixed) end-effector orientation. This enables more dexterous manipulation while maximizing the robot’s manipulability. However, implementation in a physics-based simulator remains uncertain. If time permits, we will attempt it in Genesis-AI; otherwise, results for this specific task will be presented using a non-physics-based simulator, such as Swift~\cite{swift2024}.

# Project Overview
## Robot Platform and Simulator

We will use the Tabletop Franka Emika Panda robot and simulate it in a physics-based simulator, Genesis-AI~\cite{Genesis}.

## Object Types

The object set will include cuboids, cylinders, and L-shaped rigid bodies. All objects will initially lie in random arbitrary poses on the tabletop and the goal will be to reorient them to an upright configuration.

## Simulation Assets

The default Panda robot URDF will be used. For rotational slippage, we will modify the URDF to include an additional (8th) degree of freedom as a revolute joint in between the robot's fingers. This joint will enable rotational slippage between the gripper and the object. The inverse kinematics will be adapted accordingly to solve for this added joint.

## Proposed Approach

This will be a model-based control system. The robot begins from a ready configuration. The objects' position are initialized randomly within the robot workspace. The object pose will be obtained directly from the simulator (no perception module).

A resolved motion rate controller (RMRC) will be used to track desired end-effector trajectories along the grasp path to grasp the object and screw motion path to manipulate the object, while avoiding obstacles and respecting joint position and velocity limits. Manipulability will be maximized throughout the trajectory using an optimization-based controller.

## Model Sources

All manipulated objects will be either primitive objects (boxes, cylinders) defined within Genesis-AI or CAD models from open-source repositories.
