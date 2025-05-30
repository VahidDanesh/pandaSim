"""
Resolved-Rate Motion Controller implementation.

This module provides a Resolved-Rate Motion Controller for precise end-effector velocity control.
"""
from typing import Tuple, Any, List, Dict, Optional
import numpy as np
import genesis as gs

from pytransform3d import transformations as pt, rotations as pr

from pandaSim.control.protocols import MotionController
from pandaSim.geometry.protocols import GeometryAdapter
from pandaSim.geometry.utils import convert_pose

class ResolvedRateController(MotionController):
    """
    Resolved-Rate Motion Controller for robot control.
    
    This controller computes joint velocities to achieve desired end-effector velocities,
    using nullspace projection for secondary objectives like joint limit avoidance.
    """
    
    def __init__(
        self,
        adapter: GeometryAdapter,
        gains_translation: float = 1.5,
        gains_rotation: float = 1.0,
        secondary_gain: float = 1.0,
        threshold: float = 0.001,
        end_effector_link: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Resolved-Rate Motion Controller.
        
        Args:
            adapter: Geometry adapter for interacting with the robot
            gains_translation: Proportional gain for translation control
            gains_rotation: Proportional gain for rotation control
            secondary_gain: Gain for secondary objectives (e.g., joint limit avoidance)
            threshold: Convergence threshold
            end_effector_link: End-effector link for Jacobian calculation
            config: Optional additional configuration parameters
        """
        self.adapter = adapter
        self.config = config or {}
        self.kt = gains_translation
        self.kr = gains_rotation
        self.k = np.array([self.kt, self.kt, self.kt, self.kr, self.kr, self.kr])
        self.k0 = secondary_gain
        self.threshold = threshold
        self.end_effector_link = end_effector_link
    
    def p_servo(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
        method: str = "quat"
    ) -> Tuple[np.ndarray, bool]:
        """
        Position-based servoing to calculate velocity needed to reach target pose.
        
        Args:
            current_pose: Current end-effector pose matrix (4x4)
            target_pose: Target end-effector pose matrix (4x4)
            method: Method to compute error ("twist", "rpy", or "quat")
            
        Returns:
            Tuple of (velocity_twist, arrived_flag)
        """
        current_pose = convert_pose(current_pose, output_type='t')
        target_pose = convert_pose(target_pose, output_type='t')

        eTep = np.linalg.inv(current_pose) @ target_pose
        
        # Initialize error vector
        e = np.zeros(6)
        
        if method == "twist":
            Stheta = pt.transform_log_from_transform(eTep)
            e[:6] = pt.screw_axis_from_screw_matrix(Stheta)
            
        elif method == "rpy":
            # RPY method
            e[:3] = eTep[:3, 3]
            e[3:] = pr.euler_from_matrix(eTep[:3, :3], 0, 1, 2, extrinsic=False)
            
        elif method == "quat":
            # Quaternion method
            target_pq = convert_pose(target_pose, 'pq')
            current_pq = convert_pose(current_pose, 'pq')
            
            # Position error
            error_pos = target_pq[:3] - current_pq[:3]
            
            # Orientation error using quaternions
            error_quat = gs.transform_quat_by_quat(gs.inv_quat(current_pq[3:]), target_pq[3:])
            error_rotvec = gs.quat_to_rotvec(error_quat)
            
            e = np.hstack((error_pos, error_rotvec))
            
        # Apply gains
        v = self.k * e
        
        # Check if arrived at target
        arrived = np.linalg.norm(e) < self.threshold
        
        return v, arrived
    
    def secondary_objective(
        self,
        robot: Any,
        q: np.ndarray
    ) -> np.ndarray:
        """
        Joint limit avoidance secondary objective.
        
        Args:
            robot: Robot representation
            q: Current joint angles
            
        Returns:
            Gradient for secondary objective
        """
        n = len(q)
        
        # Get joint limits from robot
        joint_limits = self.adapter.get_joint_limits(robot)
        q_min = joint_limits[0]  # Lower limits
        q_max = joint_limits[1]  # Upper limits
        
        # Create gradient for joint limit avoidance
        grad = np.zeros(n)
        for i in range(n):
            # Calculate distance from limits normalized by range
            range_i = q_max[i] - q_min[i]
            mid_i = (q_max[i] + q_min[i]) / 2
            
            # Normalized position within joint range (-1 to 1)
            norm_pos = 2 * (q[i] - mid_i) / range_i
            
            # Create repulsive gradient that increases near limits
            grad[i] = -norm_pos * (1.0 / (1.0 - np.abs(norm_pos) + 1e-6))
            
        return grad
    
    def compute_joint_velocities(
        self,
        robot: Any,
        target_pose: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute joint velocities using resolved rate control with null space optimization.
        
        Args:
            robot: Robot representation
            target_pose: Target end-effector pose matrix (4x4)
            
        Returns:
            Tuple of (joint_velocities, arrived_flag)
        """
        # Get current robot state
        q = self.adapter.get_joint_positions(robot)
        n = len(q)  # Number of joints
        
        # Get Jacobian
        J = self.adapter.compute_jacobian(robot, self.end_effector_link)
        J_pinv = np.linalg.pinv(J)
        
        # Get current end-effector pose using forward kinematics
        Te = self.adapter.forward_kinematics(robot, self.end_effector_link, q, output_type='t')
        
        # Calculate required velocity and arrival status
        ev, arrived = self.p_servo(Te, target_pose, method="quat")
        
        # Secondary objective for null space optimization
        dw_dq = self.secondary_objective(robot, q)
        q0 = self.k0 * dw_dq
        
        # Compute joint velocities with null space projection
        qd = J_pinv @ ev + (np.eye(n) - J_pinv @ J) @ q0
        
        return qd, arrived
    
    def compute_joint_velocities_quat(
        self,
        robot: Any,
        target_pose: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute joint velocities using quaternion-based resolved rate control.
        
        Args:
            robot: Robot representation
            target_pose: Target end-effector pose matrix (4x4)
            
        Returns:
            Tuple of (joint_velocities, arrived_flag)
        """
        
        # Get current end-effector pose using forward kinematics
        Te = self.adapter.forward_kinematics(robot, self.end_effector_link, output_type='t')
        
        # Calculate error and check if arrived using p_servo
        error, arrived = self.p_servo(Te, target_pose, method="quat")
        
        # Get Jacobian
        jac = self.adapter.compute_jacobian(robot, self.end_effector_link)
        
        # Add damping to avoid singularities
        damping = 1e-6
        diag = damping * np.eye(6)
        
        # Compute joint velocities using damped least squares
        qd = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
        
        return qd, arrived
    
    def execute_trajectory(
        self,
        robot: Any,
        trajectory: List[np.ndarray],
        dt: float = 0.05,
        use_control: bool = True
    ) -> None:
        """
        Execute a trajectory on the robot.
        
        Args:
            robot: Robot representation
            trajectory: List of target poses (4x4 homogeneous transformations)
            dt: Time step for simulation
            use_control: Whether to use PD control (True) or direct velocity setting (False)
        """
        for target_pose in trajectory:
            arrived = False
            while not arrived:
                # Compute joint velocities
                qd, arrived = self.compute_joint_velocities(robot, target_pose)
                
                # Apply velocities to robot
                if use_control:
                    self.adapter.control_joint_velocities(robot, qd)
                else:
                    self.adapter.set_joint_velocities(robot, qd)
                
                # Step simulation
                self.adapter.step_simulation(dt) 