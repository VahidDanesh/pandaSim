"""
Robot execution utilities for clean control loop management.

This module provides clean interfaces for executing robot control tasks.
"""
import time
import numpy as np
from typing import Any, Optional, Union, List, Literal
import logging
import roboticstoolbox as rtb
from pandaSim.control.protocols import MotionController
from pandaSim.control.config import ExecutionConfig
from pandaSim.control.data_collector import DataCollector, ExecutionResult
from pandaSim.geometry.protocols import GeometryAdapter

# Optional real robot imports
try:
    import panda_py
    from panda_py import controllers
    REAL_ROBOT_AVAILABLE = True
except ImportError:
    REAL_ROBOT_AVAILABLE = False
    logging.warning("panda_py not available - real robot execution disabled")


class RobotExecutor:
    """
    Clean robot execution manager supporting both simulation and real robot execution.
    
    Handles two main execution modes:
    1. Target approach: Move to a specific pose
    2. Velocity following: Follow velocity commands for manipulation
    """
    
    def __init__(
        self,
        controller: MotionController,
        adapter: GeometryAdapter,
        config: Optional[ExecutionConfig] = None
    ):
        """
        Initialize robot executor.
        
        Args:
            controller: Motion controller implementing MotionController protocol
            adapter: Geometry adapter for simulation
            config: Execution configuration
        """
        self.controller = controller
        self.adapter = adapter
        self.config = config or ExecutionConfig()
        self.logger = logging.getLogger(__name__)
        
    def approach_target(
        self,
        target_pose: np.ndarray,
        sim_robot: Any,
        real_robot: Optional[Any] = None,
        optimization_type: str = "qp",
        use_real_robot: bool = False
    ) -> ExecutionResult:
        """
        Execute target approach - move robot until target pose is reached.
        
        Args:
            target_pose: Target end-effector pose (4x4 matrix)
            sim_robot: Simulation robot for visualization
            real_robot: Real robot interface (required if use_real_robot=True)
            optimization_type: Controller optimization type
            use_real_robot: Whether to execute on real robot
            
        Returns:
            ExecutionResult with trajectory data and execution status
        """
        if use_real_robot and not REAL_ROBOT_AVAILABLE:
            raise RuntimeError("Real robot execution requested but panda_py not available")
        
        if use_real_robot and real_robot is None:
            raise ValueError("real_robot must be provided when use_real_robot=True")
            
        collector = DataCollector()
        
        if use_real_robot:
            return self._execute_real_approach(target_pose, sim_robot, real_robot, optimization_type, collector)
        else:
            return self._execute_sim_approach(target_pose, sim_robot, optimization_type, collector)
    
    def follow_twist_trajectory(
        self,
        twist_trajectory: List[np.ndarray],
        sim_robot: Any,
        real_robot: Optional[Any] = None,
        use_real_robot: bool = False,
        optimization_type: str = "qp"
    ) -> ExecutionResult:
        """
        Execute twist trajectory following - compute velocities in real-time based on current robot state.
        
        Args:
            twist_trajectory: List of desired twist vectors (6D: [v, omega])
            sim_robot: Simulation robot for visualization
            real_robot: Real robot interface (required if use_real_robot=True)
            use_real_robot: Whether to execute on real robot
            optimization_type: Controller optimization type
            
        Returns:
            ExecutionResult with execution data
        """
        if use_real_robot and not REAL_ROBOT_AVAILABLE:
            raise RuntimeError("Real robot execution requested but panda_py not available")
            
        if use_real_robot and real_robot is None:
            raise ValueError("real_robot must be provided when use_real_robot=True")
            
        collector = DataCollector()
        
        if use_real_robot:
            return self._execute_real_twist_trajectory(twist_trajectory, sim_robot, real_robot, optimization_type, collector)
        else:
            return self._execute_sim_twist_trajectory(twist_trajectory, sim_robot, optimization_type, collector)
    
    def follow_velocities(
        self,
        velocity_commands: List[np.ndarray],
        sim_robot: Any,
        real_robot: Optional[Any] = None,
        use_real_robot: bool = False
    ) -> ExecutionResult:
        """
        Execute velocity following - follow pre-computed velocity commands.
        
        DEPRECATED: Use follow_twist_trajectory for proper real-time control.
        This method is kept for backward compatibility with pre-computed commands.
        
        Args:
            velocity_commands: List of joint velocity commands
            sim_robot: Simulation robot for visualization
            real_robot: Real robot interface (required if use_real_robot=True)
            use_real_robot: Whether to execute on real robot
            
        Returns:
            ExecutionResult with execution data
        """
        if use_real_robot and not REAL_ROBOT_AVAILABLE:
            raise RuntimeError("Real robot execution requested but panda_py not available")
            
        if use_real_robot and real_robot is None:
            raise ValueError("real_robot must be provided when use_real_robot=True")
            
        collector = DataCollector()
        
        if use_real_robot:
            return self._execute_real_velocities(velocity_commands, sim_robot, real_robot, collector)
        else:
            return self._execute_sim_velocities(velocity_commands, sim_robot, collector)
    
    def _execute_sim_approach(
        self,
        target_pose: np.ndarray,
        sim_robot: Any,
        optimization_type: str,
        collector: DataCollector
    ) -> ExecutionResult:
        """Execute target approach in simulation only."""
        self.logger.info("Starting simulation approach to target")


        elapsed_time = 0.0
        collector.start_collection(elapsed_time)
        
        arrived = False
        step = 0
        dt = self.adapter.dt
        
        while not arrived and elapsed_time < self.config.max_runtime:
            # Compute control
            qd, arrived = self.controller.compute_joint_velocities(
                target_pose=target_pose,
                optimization_type=optimization_type
            )
            qd = self.controller.smooth_velocity_ramp(qd_cmd=qd, elapsed_time=elapsed_time, method="quintic")
            # Apply to simulation
            self.adapter.control_joint_velocities(sim_robot, qd)
            self.adapter.step_simulation(dt)
            
            # Collect data
            elapsed_time += dt
            q = self.adapter.get_joint_positions(sim_robot)
            collector.collect_step(qd, q, elapsed_time)
            
            step += 1
            
            # Safety timeout check
            if elapsed_time > self.config.safety_timeout:
                self.logger.warning("Safety timeout reached in simulation")
                break
        
        self.logger.info(f"Simulation completed: arrived={arrived}, steps={step}")
        return collector.get_result(arrived)
    
    def _execute_real_approach(
        self,
        target_pose: np.ndarray,
        sim_robot: rtb.Robot,
        real_robot: panda_py.Panda,
        optimization_type: str,
        collector: DataCollector,
        ctrl: controllers = controllers.IntegratedVelocity()
    ) -> ExecutionResult:
        """Execute target approach on real robot, keeping track of average optimization time."""
        self.logger.info("Starting real robot approach to target")
        
        real_robot.start_controller(ctrl)
        elapsed_time = ctrl.get_time()
        buffer_size = int(self.config.max_runtime * 1000)
        real_robot.enable_logging(buffer_size=buffer_size)

        # For tracking optimization time
        total_opt_time = 0.0
        opt_count = 0
        
        try:
            collector.start_collection(elapsed_time)
            arrived = False
            
            with real_robot.create_context(frequency=self.config.frequency, max_runtime=self.config.max_runtime) as ctx:
                while ctx.ok() and not arrived:
                    # Sync simulation with real robot
                    sim_robot.q[:7] = real_robot.q[:7]
                    
                    # Compute control with timing
                    import time
                    t0 = time.perf_counter()
                    qd, arrived = self.controller.compute_joint_velocities(
                        target_pose=target_pose,
                        optimization_type=optimization_type
                    )
                    t1 = time.perf_counter()
                    opt_time = t1 - t0
                    total_opt_time += opt_time
                    opt_count += 1

                    qd = self.controller.smooth_velocity_ramp(qd_cmd=qd, elapsed_time=elapsed_time, method="quintic")
                    # Apply to real robot
                    ctrl.set_control(qd[:7])
                    
                    # Update simulation for visualization
                    self.adapter.control_joint_velocities(sim_robot, qd)
                    self.adapter.step_simulation(self.adapter.dt)
                    
                    # Collect data
                    elapsed_time = ctrl.get_time()

                    q = real_robot.q[:7]
                    collector.collect_step(qd[:7], q, elapsed_time)
                    
        finally:
            real_robot.stop_controller()
            real_robot.disable_logging()
        
        avg_opt_time = total_opt_time / opt_count if opt_count > 0 else 0.0
        self.logger.info(f"Average optimization time per step: {avg_opt_time:.6f} seconds")
        self.logger.info(f"Real robot execution completed: arrived={arrived}")
        return collector.get_result(arrived)
    
    def _execute_sim_twist_trajectory(
        self,
        twist_trajectory: List[np.ndarray],
        sim_robot: Any,
        optimization_type: str,
        collector: DataCollector
    ) -> ExecutionResult:
        """Execute twist trajectory in simulation with real-time velocity computation."""
        self.logger.info(f"Starting simulation twist trajectory: {len(twist_trajectory)} twists")
        
        start_time = time.time()
        collector.start_collection(start_time)
        
        for twist in twist_trajectory:
            # Compute joint velocities based on current robot state
            qd, _ = self.controller.compute_joint_velocities(
                twist=twist,
                optimization_type=optimization_type
            )
            
            # Apply velocity command
            self.adapter.control_joint_velocities(sim_robot, qd)
            self.adapter.step_simulation(self.adapter.dt)
            
            # Collect data
            current_time = time.time()
            q = self.adapter.get_joint_positions(sim_robot)
            collector.collect_step(qd, q, current_time - start_time)
            
        self.logger.info("Simulation twist trajectory completed")
        return collector.get_result(True)  # Twist following always "converges"
    
    def _execute_real_twist_trajectory(
        self,
        twist_trajectory: List[np.ndarray],
        sim_robot: Any,
        real_robot: Any,
        optimization_type: str,
        collector: DataCollector
    ) -> ExecutionResult:
        """Execute twist trajectory on real robot with real-time velocity computation."""
        self.logger.info(f"Starting real robot twist trajectory: {len(twist_trajectory)} twists")
        
        # Ensure robot is unlocked
        if hasattr(real_robot, 'unlock'):
            real_robot.unlock()
            
        # Setup controller
        ctrl = controllers.IntegratedVelocity()
        real_robot.start_controller(ctrl)
        real_robot.enable_logging(len(twist_trajectory) + 1000)
        
        try:
            start_time = time.time()
            collector.start_collection(start_time)
            
            with real_robot.create_context(frequency=self.config.frequency) as ctx:
                for twist in twist_trajectory:
                    if not ctx.ok():
                        break
                        
                    # Sync simulation
                    sim_robot.q[:sim_robot.n] = real_robot.q[:sim_robot.n]
                    
                    # Compute joint velocities based on current robot state
                    qd, _ = self.controller.compute_joint_velocities(
                        twist=twist,
                        optimization_type=optimization_type
                    )
                    
                    # Apply velocity command
                    ctrl.set_control(qd[:sim_robot.n])
                    
                    # Update simulation for visualization
                    self.adapter.control_joint_velocities(sim_robot, qd)
                    self.adapter.step_simulation(self.config.sim_dt)
                    
                    # Collect data
                    current_time = ctrl.get_time()
                    q = real_robot.q[:sim_robot.n]
                    collector.collect_step(qd[:sim_robot.n], q, current_time)
                    
        finally:
            real_robot.stop_controller()
            real_robot.disable_logging()
            
        self.logger.info("Real robot twist trajectory completed")
        return collector.get_result(True)  # Twist following always "converges"
    
    def _execute_sim_velocities(
        self,
        velocity_commands: List[np.ndarray],
        sim_robot: Any,
        collector: DataCollector
    ) -> ExecutionResult:
        """Execute velocity commands in simulation."""
        self.logger.info(f"Starting simulation velocity following: {len(velocity_commands)} commands")
        
        start_time = time.time()
        collector.start_collection(start_time)
        
        for i, qd_cmd in enumerate(velocity_commands):
            # Apply velocity command
            self.adapter.control_joint_velocities(sim_robot, qd_cmd)
            self.adapter.step_simulation(self.config.sim_dt)
            
            # Collect data
            current_time = time.time()
            q = self.adapter.get_joint_positions(sim_robot)
            collector.collect_step(qd_cmd, q, current_time - start_time)
            
        self.logger.info("Simulation velocity following completed")
        return collector.get_result(True)  # Velocity following always "converges"
    
    def _execute_real_velocities(
        self,
        velocity_commands: List[np.ndarray],
        sim_robot: Any,
        real_robot: Any,
        collector: DataCollector
    ) -> ExecutionResult:
        """Execute velocity commands on real robot."""
        self.logger.info(f"Starting real robot velocity following: {len(velocity_commands)} commands")
        
            
        # Setup controller
        ctrl = controllers.IntegratedVelocity()
        real_robot.start_controller(ctrl)
        real_robot.enable_logging(len(velocity_commands) + 1000)
        
        try:
            start_time = time.time()
            collector.start_collection(start_time)
            
            with real_robot.create_context(frequency=self.config.frequency) as ctx:
                for qd_cmd in velocity_commands:
                    if ctx.ok():
                        continue
                        
                    # Sync simulation
                    sim_robot.q[:sim_robot.n] = real_robot.q[:sim_robot.n]
                    
                    # Apply velocity command
                    ctrl.set_control(qd_cmd[:sim_robot.n])
                    
                    # Update simulation for visualization
                    self.adapter.control_joint_velocities(sim_robot, qd_cmd)
                    self.adapter.step_simulation(self.config.sim_dt)
                    
                    # Collect data
                    current_time = ctrl.get_time()
                    q = real_robot.q[:sim_robot.n]
                    collector.collect_step(qd_cmd[:sim_robot.n], q, current_time)
                    
        finally:
            real_robot.stop_controller()
            real_robot.disable_logging()
            
        self.logger.info("Real robot velocity following completed")
        return collector.get_result(True)  # Velocity following always "converges"
