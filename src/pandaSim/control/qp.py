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
import spatialmath as sm
from spatialmath import SE3
import modern_robotics as mr

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
        robot: Optional[Any] = None,
        end_effector_link: Optional[Any] = None,
        gains_translation: float = 5.0,
        gains_rotation: float = 2.0,
        threshold: float = 0.001,
        lambda_q: float = 0.5,
        lambda_m: float = 0.1,
        lambda_j: float = 0.1,
        ps: float = 0.05,
        pi: float = 0.3,
        eta: float = 1.0,
        solver: str = 'quadprog',
        T: float = 1.0,
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
            lambda_m: Manipulability maximization weight (λ_m)
            lambda_j: Joint limit minimization weight (λ_j)
            ps: Joint limit stopping distance (ρₛ)
            pi: Joint limit influence distance (ρᵢ)
            eta: Joint limit damper gain (η)
            solver: QP solver to use
            T: Ramp-up time in seconds
        """
        self.adapter = adapter
        self.robot = robot
        self.end_effector_link = end_effector_link
        self.kt = gains_translation
        self.kr = gains_rotation
        self.k = np.array([self.kt, self.kt, self.kt, self.kr, self.kr, self.kr])
        self.threshold = threshold
        
        # QP parameters
        self.lambda_q = lambda_q
        self.lambda_m = lambda_m
        self.lambda_j = lambda_j
        self.ps = ps  # ρₛ
        self.pi = pi  # ρᵢ
        self.eta = eta  # η
        self.solver = solver
        self.T = T
    
    def p_servo(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
        method: str = "twist"
    ) -> Tuple[np.ndarray, bool]:
        """
        Position-based servoing using robotics toolbox implementation.
        
        Args:
            current_pose: Current end-effector pose matrix (4x4)
            target_pose: Target end-effector pose matrix (4x4)
            method: Method to compute error ("twist" or "rpy" or "angle-axis")
            
        Returns:
            Tuple of (velocity_twist, arrived_flag)
        """
        # if self.k is scalar, convert it to a matrix
        if np.isscalar(self.k):
            k = self.k * np.eye(6)
        else:
            k = np.diag(self.k)

        if isinstance(current_pose, SE3):
            current_pose = current_pose.A
        if isinstance(target_pose, SE3):
            target_pose = target_pose.A
            
        if method.startswith("t"):
            bTd = mr.TransInv(current_pose) @ target_pose
            ang_axis = mr.se3ToVec(mr.MatrixLog6(bTd)) 
            axis_ang = np.concatenate([ang_axis[3:], ang_axis[:3]])
            v = k @ axis_ang
            arrived = True if np.sum(np.abs(axis_ang)) < self.threshold else False

        else:
            # Use RTB p_servo function
            v, arrived = rtb.p_servo(
                current_pose, target_pose, 
                gain=self.k, 
                threshold=self.threshold, 
                method=method
            )
            
        return v, arrived
    
    def smooth_velocity_ramp(self, qd_cmd: np.ndarray, elapsed_time: float, method: str = 'quintic') -> np.ndarray:
        """
        Compute smooth joint velocity command using specified time scaling method.
        
        Args:
            q_cmd (np.ndarray): Desired joint velocity command from QP solver.
            q_act (np.ndarray): Current joint velocity.
            T (float): Ramp-up time in seconds.
            t (float): Current time since ramp start.
            dt (float): Control cycle time in seconds.
            method (str): Scaling method, either 'quintic' or 'cosine'.
        
        Returns:
            np.ndarray: Scaled joint velocity command.
        """
        # Compute scaling factor based on method
        T = self.T
        if elapsed_time < 0:
            s = 0.0
        elif elapsed_time <= T:
            tau = elapsed_time / T
            if method == 'quintic':
                s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            elif method == 'cosine':
                s = 0.5 * (1 - np.cos(np.pi * tau))
            else:
                raise ValueError("Method must be 'quintic' or 'cosine'")
        else:
            s = 1.0
        
        qd = self.adapter.get_joint_velocities(self.robot)
        # Compute scaled command
        if np.linalg.norm(qd) < 1e-6:  # Assume zero if small
            qd_scaled = s * qd_cmd
        else:
            # qd_scaled = (1 - s) * qd + s * qd_cmd  # continuous with current joint velocity
            qd_scaled = s * qd_cmd
        
        return qd_scaled

    def joint_velocity_damper(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Formulates an inequality constraint which, when optimised for will
        make it impossible for the robot to run into joint limits.
        
        
        
        Args:
            robot: Robot representation
            
        Returns:
            Tuple of (A_in, b_in) for inequality constraint A_in * qd <= b_in
            A_in: n x n matrix, b_in: n-dimensional vector
        """
        n = self.adapter.get_dof(self.robot)
        q = self.adapter.get_joint_positions(self.robot)  # Current joint positions from robot
        lower_limits, upper_limits = self.adapter.get_joint_limits(self.robot)
        
        # Initialize A_in as n×n matrix and b_in as n-dimensional vector
        A_in = np.zeros((n, n))
        b_in = np.zeros(n)
        
        for i in range(n):
            # Check if within influence distance of lower limit
            if q[i] - lower_limits[i] <= self.pi:
                b_in[i] = -self.eta * (((lower_limits[i] - q[i]) + self.ps) / (self.pi - self.ps))
                A_in[i, i] = -1
                
            # Check if within influence distance of upper limit  
            if upper_limits[i] - q[i] <= self.pi:
                b_in[i] = self.eta * ((upper_limits[i] - q[i]) - self.ps) / (self.pi - self.ps)
                A_in[i, i] = 1
                
        return A_in, b_in
    
    def compute_joint_velocities(
        self,
        target_pose: Optional[np.ndarray] = None, 
        twist: Optional[np.ndarray] = None,
        optimization_type: Optional[str] = "qp"
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute joint velocities using QP formulation:
        
        min (1/2) xᵀ Q x + cᵀ x
        s.t. A_in x <= b_in, A_eq x = b_eq, lb <= x <= ub
        
        Args:
            robot: Robot representation
            target_pose: Target end-effector pose matrix (4x4)
            twist: End-effector desired twist, either target_pose or twist should be provided, (6,), in format of (v, omega)
        Returns:
            Tuple of (joint_velocities, arrived_flag)
        """
        if target_pose is None and twist is None:
            raise ValueError("Either target_pose or twist must be provided")
        
        if target_pose is not None and twist is not None:
            raise ValueError("Only one of target_pose or twist can be provided")
        

        # Get current robot state
        q = self.adapter.get_joint_positions(self.robot)
        # Get current joint velocities
        qd = self.adapter.get_joint_velocities(self.robot)

        n = self.adapter.get_dof(self.robot)
        
        # Get current end-effector pose
        Te = self.adapter.forward_kinematics(self.robot, self.end_effector_link, q, output_type='t')

        # Get Jacobian J_e
        Je = self.robot.jacobe(q, end=self.end_effector_link)
        He = self.robot.hessiane(Je=Je, end=self.end_effector_link)

       
        
        # Calculate required end-effector velocity V_b using RTB p_servo
        if twist is None:
            v_b, arrived = self.p_servo(Te, target_pose, method="t") # (v, omega)
        else:
            v_b = twist.reshape((6,)) 
            v_b_current = Je @ qd
            arrived = np.linalg.norm(v_b - v_b_current) < self.threshold


        
        # Get manipulability Jacobian J_m using RTB jacobm
        try:
            # Try to get RTB robot object from adapter
            if hasattr(self.robot, 'jacobm'):
                J_m = self.robot.jacobm(J=Je, H=He, end=self.end_effector_link, axes='all')
            else:
                print("Warning: Could not compute manipulability Jacobian, using zero")
                J_m = np.zeros(n)
        except:
            # Final fallback: zero manipulability gradient
            print("Warning: Could not compute manipulability Jacobian, using zero")
            J_m = np.zeros(n)
        
        ### QP Formulation ###
        
        # Quadratic term: Q = λ * I_n
        Q = self.lambda_q * np.eye(n)


        if optimization_type.startswith("q"):
            # Linear term: c = -λ_m * J_m (negative for maximization)
            c = -self.lambda_m * J_m.reshape((n,))
        elif optimization_type.startswith("j"):
            l_qlim, u_qlim = self.adapter.get_joint_limits(self.robot)
            mid_q = (l_qlim + u_qlim) / 2
            c = self.lambda_j * (q - mid_q) - self.lambda_m * J_m.reshape((n,))
        else:
            raise ValueError(f"Invalid optimization type: {optimization_type}")
        
        
        # Equality constraint: J_b * x = V_b
        A_eq = Je
        b_eq = v_b
        
        # Inequality constraints: joint velocity dampers
        A_in, b_in = self.joint_velocity_damper()
        
        # Constraints: x_min ≤ x ≤ x_max
        lb, ub = self.adapter.get_joint_velocity_limits(self.robot)


        qd = qp.solve_qp(
            P=Q, q=c, G=A_in, h=b_in, A=A_eq, b=b_eq,
            lb=lb, ub=ub, solver=self.solver, initvals=qd
        )


        if qd is None:
            # Fallback to pseudoinverse if QP fails
            print("QP solver failed, try different hyperparameters")
            return None, False
        
        return qd, arrived
    
    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        dt: float = 0.01,
        use_control: bool = True
    ) -> None:
        """
        Execute a trajectory on the robot using QP control.
        
        Args:
            trajectory: List of target poses (4x4 homogeneous transformations)
            dt: Time step for simulation
            use_control: Whether to use PD control (True) or direct velocity setting (False)
        """
        for target_pose in trajectory:
            arrived = False
            print(target_pose)
            while not arrived:
                # Compute joint velocities using QP
                qd, arrived = self.compute_joint_velocities(target_pose)
                
                # Apply velocities to robot
                if use_control:
                    self.adapter.control_joint_velocities(self.robot, qd)
                else:
                    self.adapter.set_joint_velocities(self.robot, qd)
                
                # Step simulation
                self.adapter.step_simulation(dt)
                