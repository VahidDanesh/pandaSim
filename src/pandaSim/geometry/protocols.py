"""
Protocols for geometry adapters.

These protocols define the interface that any geometry backend adapter must implement.
"""
from typing import Protocol, Any, List, Tuple, Optional, Dict, Union, runtime_checkable
import numpy as np
import torch
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
    
    def add_primitive(self, primitive_type: str, **params) -> Dict:
        """
        Add a primitive shape to the scene.
        
        Args:
            primitive_type: Type of primitive shape ("box", "cylinder", "sphere", "plane")
            **params: Shape-specific parameters including surface properties
        
        Returns:
            Dictionary containing the entity information
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
    def get_obb(self, obj: Union[Dict, Any]) -> Dict:
        """
        Calculate oriented bounding box for an object.
        
        Args:
            obj: Object representation or direct entity
        """
        ...
    def get_bbox(self, obj: Union[Dict, Any]) -> Dict:
        """
        Calculate axis-aligned bounding box for an object.
        
        Args:
            obj: Object representation or direct entity
        
        Returns:
            Dictionary containing bounding box information
        """
        ...
    
    def get_pose(self, obj: Union[Dict, Any], output_type: Optional[str] = 'pq') -> np.ndarray:
        """
        Get the pose of the object.

        Args:
            obj: Object representation or direct entity
            output_type: Desired output format ('pq', 'transform', 'dual_quaternion', etc.)
        Returns:
            Pose of the object in the requested format
        """
        ...
    
    def set_pose(self, obj: Union[Dict, Any], pose: Union[tuple, np.ndarray, torch.Tensor]) -> None:
        """
        Set the pose of the object.

        Args:
            obj: Object representation or direct entity
            pose: Pose to set, can be (4, 4) transformation matrix, (7,) pq or (8,) dq
        """
        ...
    
    def get_size(self, obj: Union[Dict, Any]) -> np.ndarray:
        """
        Get the size of the bbox for the given object, represented in object's coordinate frame.

        Args:
            obj: Object representation or direct entity
        Returns:
            Size of the bbox, in (x, y, z) order of it's own coordinate frame.
        """
        ...
    
    def to(self, transformation: Any, output_type: Optional[str] = 'pq') -> np.ndarray:
        """
        Convert input to the given output_type.
        
        Args:
            input: Can be one of the following representations
            output_type: Desired output format
            
        Returns:
            Converted representation in the requested format
        """
        ...
    
    def transform(self, 
                  obj: Union[Dict, Any], 
                  transformation: Union[tuple, np.ndarray, torch.Tensor], 
                  apply: bool = False,
                  output_type: Optional[str] = 't'
                  ) -> Dict:
        """
        Apply transformation to the geometry.
        
        Args:
            obj: The geometry representation
            transformation: Transformation to apply
            apply: Whether to apply the transformation to the object
            output_type: Format for the returned transformation
            
        Returns:
            Transformed pose of the object in the requested format
        """
        ...
    
    # Robot control methods
    
    def get_dof(self, robot: Any) -> int:
        """
        Get the number of degrees of freedom of the robot.
        """
        ...
    
    def get_joint_velocity_limits(self, robot: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint velocity limits of the robot.
        """
        ...

    def get_joint_positions(self, robot: Any) -> np.ndarray:
        """
        Get current joint positions of the robot.
        
        Args:
            robot: The robot representation
            
        Returns:
            Array of joint positions
        """
        ...
    
    def get_joint_velocities(self, robot: Any) -> np.ndarray:
        """
        Get current joint velocities of the robot.
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
    
    def compute_jacobian(self, robot: Any, link: Any, base: Optional[bool]) -> np.ndarray:
        """
        Compute the Jacobian matrix for the robot.
        
        Args:
            robot: The robot representation
            link: End-effector link
            base: Whether to compute the Jacobian for the base frame (default: False)
        Returns:
            Jacobian matrix
        """
        ...
    
    def forward_kinematics(self, 
                           robot: Any, 
                           link: Any = None, 
                           q: Optional[np.ndarray] = None,
                           output_type: Optional[str] = 'dq') -> np.ndarray:
        """
        Compute forward kinematics for the robot.
        
        Args:
            robot: The robot representation
            link: End-effector link (optional)
            q: Joint positions (optional)
            output_type: Type of output ('t', 'dq', 'pq')
            
        Returns:
            End-effector or all links pose as requested format
        """
        ...
    
    def step_simulation(self, time: float) -> None:
        """
        Step the simulation forward by the specified time.
        
        Args:
            time: wallclock time to step the simulation
        """
        ...