"""
Protocols for geometry adapters.

These protocols define the interface that any geometry backend adapter must implement.
"""
from typing import Protocol, Any, List, Tuple, Optional, runtime_checkable
import numpy as np
import genesis as gs

@runtime_checkable
class GeometryAdapter(Protocol):
    """
    Protocol for geometry backend adapters.
    
    Any adapter must implement these methods to be compatible with the system.
    """
    
    def load(self, file_path: str) -> Any:
        """
        Load geometry data from file.
        
        Args:
            file_path: Path to the geometry file (mesh, point cloud, etc.)
            
        Returns:
            The loaded geometry representation
        """
        ...
    
    def get_vertices(self, obj: Any) -> np.ndarray:
        """
        Extract vertices from the geometry.
        
        Args:
            obj: The geometry representation
            
        Returns:
            Numpy array of vertices (Nx3)
        """
        ...
    
    def get_faces(self, obj: Any) -> Optional[np.ndarray]:
        """
        Extract faces from the geometry, if available.
        
        Args:
            obj: The geometry representation
            
        Returns:
            Numpy array of face indices or None if not applicable
        """
        ...
    
    def transform(self, obj: Any, transformation: np.ndarray) -> Any:
        """
        Apply transformation to the geometry.
        
        Args:
            obj: The geometry representation
            transformation: 4x4 transformation matrix
            
        Returns:
            Transformed geometry
        """
        ...
    
    # Robot control methods
    
    def get_joint_positions(self, robot: Any) -> np.ndarray:
        """
        Get current joint positions of the robot.
        
        Args:
            robot: The robot representation
            
        Returns:
            Array of joint positions
        """
        ...
    
    def set_joint_positions(self, robot: Any, positions: np.ndarray) -> None:
        """
        Set joint positions of the robot directly (without control).
        
        Args:
            robot: The robot representation
            positions: Array of joint positions
        """
        ...
    
    def control_joint_positions(self, robot: Any, positions: np.ndarray) -> None:
        """
        Control joint positions of the robot using PD controller.
        
        Args:
            robot: The robot representation
            positions: Array of joint positions (target)
        """
        ...
    
    def set_joint_velocities(self, robot: Any, velocities: np.ndarray) -> None:
        """
        Set joint velocities of the robot directly (without control).
        
        Args:
            robot: The robot representation
            velocities: Array of joint velocities
        """
        ...
    
    def control_joint_velocities(self, robot: Any, velocities: np.ndarray) -> None:
        """
        Control joint velocities of the robot using PD controller.
        
        Args:
            robot: The robot representation
            velocities: Array of joint velocities (target)
        """
        ...
    
    def get_joint_limits(self, robot: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint limits of the robot.
        
        Args:
            robot: The robot representation
            
        Returns:
            Tuple of (lower_limits, upper_limits)
        """
        ...
    
    def compute_jacobian(self, robot: Any, link: Any) -> np.ndarray:
        """
        Compute the Jacobian matrix for the robot.
        
        Args:
            robot: The robot representation
            link: End-effector link
            
        Returns:
            Jacobian matrix
        """
        ...
    
    def compute_forward_kinematics(self, robot: Any, q: Optional[np.ndarray] = None, link: Any = None) -> np.ndarray:
        """
        Compute forward kinematics for the robot.
        
        Args:
            robot: The robot representation
            q: Joint positions (optional)
            link: End-effector link (optional)
            
        Returns:
            End-effector pose as 4x4 homogeneous transformation matrix
        """
        ...
    
    def step_simulation(self, dt: float = None) -> None:
        """
        Step the simulation forward by dt seconds.
        
        Args:
            dt: Time step in seconds (optional, uses scene default if None)
        """
        ...