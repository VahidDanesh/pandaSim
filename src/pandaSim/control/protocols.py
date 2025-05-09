"""
Protocols for motion controllers.

These protocols define the interface that any motion controller must implement.
"""
from typing import Protocol, Any, Tuple, List, runtime_checkable
import numpy as np

from pandaSim.geometry.protocols import GeometryAdapter


@runtime_checkable
class MotionController(Protocol):
    """
    Protocol for motion controllers.
    
    Any motion controller must implement these methods to be compatible with the system.
    """
    
    def compute_joint_velocities(
        self,
        robot: Any,
        target_pose: np.ndarray,
        adapter: GeometryAdapter
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute joint velocities to achieve a target pose.
        
        Args:
            robot: The robot representation
            target_pose: The target end-effector pose (4x4 homogeneous transformation)
            adapter: The geometry adapter to use for accessing the geometry
            
        Returns:
            Tuple containing:
            - Joint velocities (n-dimensional vector)
            - Boolean flag indicating if the target is reached
        """
        ...
    
    def execute_trajectory(
        self,
        robot: Any,
        trajectory: List[np.ndarray],
        adapter: GeometryAdapter,
        dt: float = 0.05,
        use_control: bool = True
    ) -> None:
        """
        Execute a trajectory on the robot.
        
        Args:
            robot: The robot representation
            trajectory: List of target poses (4x4 homogeneous transformations)
            adapter: The geometry adapter to use for accessing the geometry
            dt: Time step for simulation
            use_control: Whether to use PD control (True) or direct velocity setting (False)
        """
        ... 