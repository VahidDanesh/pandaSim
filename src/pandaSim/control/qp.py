"""
Quadratic Programming Motion Controller implementation.

This module provides a QP-based motion controller implementing the formulation:
min (1/2) xᵀ Q x + cᵀ x
subject to: A_in x <= b_in, A_eq x = b_eq, lb <= x <= ub
where x = (q̇, θ̇) includes joint and virtual finger joint velocities.
"""
from typing import Tuple, Any, List, Dict, Optional, Union
import numpy as np
import qpsolvers as qp
import roboticstoolbox as rtb
from spatialmath import SE3

from pandaSim.control.protocols import MotionController
from pandaSim.geometry.protocols import GeometryAdapter
from pandaSim.geometry.utils import convert_pose


class QPController(MotionController):
    """
    Quadratic Programming Motion Controller.
    
    Implements the QP formulation with manipulability maximization and joint limit avoidance.
    """
    
    def __init__(
        self,
        adapter: GeometryAdapter,
        gains_translation: float = 5.0,
        gains_rotation: float = 2.0,
        threshold: float = 0.001,
        end_effector_link: Optional[Any] = None,
        lambda_q: float = 0.5,
        lambda_m: float = 0.1,
        ps: float = 0.05,
        pi: float = 0.3,
        eta: float = 1.0,
        solver: str = 'quadprog'
    ):
        """
        Initialize QP Motion Controller.
        
        Args:
            adapter: Geometry adapter for interacting with the robot
            gains_translation: Proportional gain for translation control
            gains_rotation: Proportional gain for rotation control
            threshold: Convergence threshold
            end_effector_link: End-effector link for Jacobian calculation
            lambda_q: Joint velocity minimization weight (λ)
            lambda_m: Manipulability maximization weight
            ps: Joint limit stopping distance (ρₛ)
            pi: Joint limit influence distance (ρᵢ)
            eta: Joint limit damper gain (η)
            solver: QP solver to use
        """
        self.adapter = adapter
        self.kt = gains_translation
        self.kr = gains_rotation
        self.k = np.array([self.kt, self.kt, self.kt, self.kr, self.kr, self.kr])
        self.threshold = threshold
        self.end_effector_link = end_effector_link
        
        # QP parameters
        self.lambda_q = lambda_q
        self.lambda_m = lambda_m
        self.ps = ps  # ρₛ
        self.pi = pi  # ρᵢ
        self.eta = eta  # η
        self.solver = solver
    
    def p_servo(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
        method: str = "angle-axis"
    ) -> Tuple[np.ndarray, bool]:
        """
        Position-based servoing using robotics toolbox implementation.
        
        Args:
            current_pose: Current end-effector pose matrix (4x4)
            target_pose: Target end-effector pose matrix (4x4)
            method: Method to compute error ("rpy" or "angle-axis")
            
        Returns:
            Tuple of (velocity_twist, arrived_flag)
        """
        # Use RTB p_servo function
        v, arrived = rtb.p_servo(
            current_pose, target_pose, 
            gain=self.k, 
            threshold=self.threshold, 
            method=method
        )
        
        return v, arrived
    
    def joint_velocity_damper(self, robot: Any, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create joint limit avoidance constraints.
        
        Implements: xᵢ ≤ η (ρᵢ - ρₛ)/(ρᵢ - ρₛ) for i = 1...n
        
        Args:
            robot: Robot representation
            q: Current joint positions
            
        Returns:
            Tuple of (A_in, b_in) for inequality constraint A_in * qd <= b_in
        """
        n = len(q)
        joint_limits = self.adapter.get_joint_limits(robot)
        q_min, q_max = joint_limits[0], joint_limits[1]
        
        A_in_list = []
        b_in_list = []
        
        for i in range(n):
            # Distance to lower limit
            rho_lower = q[i] - q_min[i]
            # Distance to upper limit  
            rho_upper = q_max[i] - q[i]
            
            # Check if within influence distance of lower limit
            if rho_lower <= self.pi:
                # For lower limit, we want xᵢ ≥ -η(ρ-ρₛ)/(ρᵢ-ρₛ)
                # Rearranged: -xᵢ ≤ η(ρ-ρₛ)/(ρᵢ-ρₛ)
                constraint_row = np.zeros(n)
                constraint_row[i] = -1
                A_in_list.append(constraint_row)
                b_in_list.append(self.eta * (rho_lower - self.ps) / (self.pi - self.ps))
            
            # Check if within influence distance of upper limit
            if rho_upper <= self.pi:
                # For upper limit: xᵢ ≤ η(ρ-ρₛ)/(ρᵢ-ρₛ)
                constraint_row = np.zeros(n)
                constraint_row[i] = 1
                A_in_list.append(constraint_row)
                b_in_list.append(self.eta * (rho_upper - self.ps) / (self.pi - self.ps))
        
        if A_in_list:
            A_in = np.array(A_in_list)
            b_in = np.array(b_in_list)
        else:
            A_in = np.zeros((0, n))
            b_in = np.zeros(0)
                
        return A_in, b_in
    
    def compute_joint_velocities(
        self,
        robot: Any,
        target_pose: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute joint velocities using QP formulation:
        
        min (1/2) xᵀ Q x + cᵀ x
        s.t. A_in x <= b_in, A_eq x = b_eq, lb <= x <= ub
        
        Args:
            robot: Robot representation
            target_pose: Target end-effector pose matrix (4x4)
            
        Returns:
            Tuple of (joint_velocities, arrived_flag)
        """
        # Get current robot state
        q = self.adapter.get_joint_positions(robot)
        n = len(q)
        
        # Get current end-effector pose
        Te = self.adapter.forward_kinematics(robot, self.end_effector_link, q, output_type='t')
        
        # Calculate required end-effector velocity V_b using RTB p_servo
        v_b, arrived = self.p_servo(Te, target_pose)
        
        # Get Jacobian J_b
        J_b = self.adapter.compute_jacobian(robot, self.end_effector_link)
        
        # Get manipulability Jacobian J_m using RTB jacobm
        try:
            # Try to get RTB robot object from adapter
            if hasattr(robot, 'jacobm'):
                J_m = robot.jacobm(q, end=self.end_effector_link)
            else:
                # Fallback: try to call jacobm through adapter
                J_m = self.adapter.compute_manipulability_jacobian(robot, self.end_effector_link)
        except:
            # Final fallback: zero manipulability gradient
            print("Warning: Could not compute manipulability Jacobian, using zero")
            J_m = np.zeros(n)
        
        ### QP Formulation ###
        
        # Quadratic term: Q = λ * I_n
        Q = self.lambda_q * np.eye(n)
        
        # Linear term: c = -λ_m * J_m (negative for maximization)
        c = -self.lambda_m * J_m
        
        # Equality constraint: J_b * x = V_b
        A_eq = J_b
        b_eq = v_b
        
        # Inequality constraints: joint velocity dampers
        A_in, b_in = self.joint_velocity_damper(robot, q)
        
        # Box constraints: x_min ≤ x ≤ x_max
        qd_max = 2.0  # rad/s - can be made configurable
        lb = -qd_max * np.ones(n)
        ub = qd_max * np.ones(n)
        
        # Solve QP
        try:
            qd = qp.solve_qp(
                Q, c, A_in, b_in, A_eq, b_eq,
                lb=lb, ub=ub, solver=self.solver
            )
            
            if qd is None:
                # Fallback to pseudoinverse if QP fails
                print("QP solver failed, using pseudoinverse fallback")
                qd = np.linalg.pinv(J_b) @ v_b
                
        except Exception as e:
            print(f"QP solver error: {e}, using pseudoinverse fallback")
            qd = np.linalg.pinv(J_b) @ v_b
            
        return qd, arrived
    
    def execute_trajectory(
        self,
        robot: Any,
        trajectory: List[np.ndarray],
        dt: float = 0.01,
        use_control: bool = True
    ) -> None:
        """
        Execute a trajectory on the robot using QP control.
        
        Args:
            robot: Robot representation
            trajectory: List of target poses (4x4 homogeneous transformations)
            dt: Time step for simulation
            use_control: Whether to use PD control (True) or direct velocity setting (False)
        """
        for target_pose in trajectory:
            arrived = False
            
            while not arrived:
                # Compute joint velocities using QP
                qd, arrived = self.compute_joint_velocities(robot, target_pose)
                
                # Apply velocities to robot
                if use_control:
                    self.adapter.control_joint_velocities(robot, qd)
                else:
                    self.adapter.set_joint_velocities(robot, qd)
                
                # Step simulation
                self.adapter.step_simulation(dt)
                