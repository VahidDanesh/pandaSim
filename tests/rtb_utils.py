import numpy as np
import spatialmath as sm
from pandaSim.geometry.rtb_adapter import RoboticsToolboxAdapter
from pandaSim.geometry.utils import create_virtual_panda
from pandaSim.control.resolved_rate import ResolvedRateController

class RTBUtils:
    """Utility class for Robotics Toolbox Adapter and Resolved Rate Controller tests."""
    def __init__(self, urdf_path=None, controller_gains=None):
        self.adapter = RoboticsToolboxAdapter({"realtime": True, "rate": 100})
        self.robot = create_virtual_panda(urdf_path=urdf_path)
        self.robot_dict = {"id": "panda", "entity": self.robot}
        self.adapter.entities["panda"] = self.robot
        self.adapter.env.add(self.robot)
        self.ee_link = 'panda_finger_virtual'  # End-effector link name
        gains = controller_gains or {}
        self.controller = ResolvedRateController(
            adapter=self.adapter,
            end_effector_link=self.ee_link,
            **gains
        )

    def add_box(self, pos, scale=(0.1, 0.07, 0.03), color='blue'):
        box = self.adapter.add_primitive(
            "box",
            scale=scale,
            color=color,
            T=sm.SE3(*pos)
        )
        return box

    def move_to_pose(self, target_pose, tol=1e-3, max_steps=200, dt=0.05, verbose=False):
        """
        Move the robot end-effector to the target pose using resolved-rate control.
        target_pose: 4x4 SE3 or (7,) pq pose
        """
        if isinstance(target_pose, (list, tuple, np.ndarray)) and np.array(target_pose).shape == (7,):
            # Convert pq to SE3
            target_pose = sm.SE3(*target_pose[:3]) * sm.SE3.Quaternion(target_pose[3:])
        elif isinstance(target_pose, (list, tuple, np.ndarray)) and np.array(target_pose).shape == (4,4):
            target_pose = sm.SE3(target_pose)
        arrived = False
        steps = 0
        while not arrived and steps < max_steps:
            qd, arrived = self.controller.compute_joint_velocities(self.robot_dict, target_pose.A)
            self.adapter.control_joint_velocities(self.robot_dict, qd)
            self.adapter.step_simulation(dt)
            steps += 1
            if verbose:
                print(f"Step {steps}: arrived={arrived}, qd={qd}")
        return arrived, steps

    def get_ee_pose(self):
        return self.adapter.forward_kinematics(self.robot_dict, self.ee_link, output_type='t')

    def reset(self):
        self.robot.q = self.robot.qr
        self.adapter.set_joint_positions(self.robot_dict, self.robot.qr)
        self.adapter.step_simulation(0.1) 