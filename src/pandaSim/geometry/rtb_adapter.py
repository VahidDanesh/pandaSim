"""
Robotics Toolbox adapter implementation.

This module provides an adapter for the Robotics Toolbox geometry backend.
"""
from typing import Any, Optional, Dict, Tuple, Union
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
from swift import Swift
from pathlib import Path
import trimesh
from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)
from pandaSim.geometry.utils import convert_pose

class RoboticsToolboxAdapter:
    """
    Adapter for Robotics Toolbox geometry backend.
    
    Implements the GeometryAdapter protocol for Robotics Toolbox.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Robotics Toolbox adapter.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Create Swift environment for visualization
        self.env = Swift()
        self.env.launch(
            realtime=self.config.get("realtime", True),
            rate=self.config.get("rate", 100),
            browser=self.config.get("browser", "notebook")
        )
        
        # Store loaded entities
        self.entities = {}
        
    @property
    def get_env(self):
        """Get the underlying Swift environment for direct access"""
        return self.env
    
    @property
    def dt(self):
        """Get the simulation time step"""
        return 1/self.env.rate
    
    def load(self, file_path: str) -> Any:
        """
        Load geometry data from file using Robotics Toolbox.
        
        Args:
            file_path: Path to the geometry file, robot model or object file
            
        Returns:
            Robotics Toolbox representation of the geometry
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.urdf':
            # Load URDF robot model
            robot = rtb.Robot.URDF(str(file_path))
            self.env.add(robot)
            
        elif file_path.suffix.lower() in ['.obj', '.stl', '.ply']:
            # Load mesh file using spatialgeometry
            mesh = sg.Mesh(filename=str(file_path))
            self.env.add(mesh)
            robot = mesh
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Store entity with an ID
        entity_id = str(len(self.entities))
        self.entities[entity_id] = robot
        
        return {
            "id": entity_id,
            "path": str(file_path),
            "entity": robot
        }
    
    def add_primitive(self, primitive_type: str, **params) -> Dict:
        """
        Add a primitive shape to the scene.
        
        Args:
            primitive_type: Type of primitive shape ("box", "cylinder", "sphere")
            **params: Shape-specific parameters including surface properties
        
        Returns:
            Dictionary containing the entity information
        """
        primitive_type = primitive_type.lower()
        
        # Extract pose parameters
        pos = params.pop('pos', [0, 0, 0])
        T = params.pop('T', None)
        
        # Create pose
        if T is not None:
            pose = T
        else:
            pose = sm.SE3(pos)
        
        if primitive_type == "box":
            # Extract box parameters
            scale = params.get('scale', [1, 1, 1])
            color = params.get('color', 'blue')
            
            entity = sg.Cuboid(scale=scale, color=color)
            entity.T = pose
            
        elif primitive_type == "cylinder":
            # Extract cylinder parameters  
            radius = params.get('radius', 0.5)
            length = params.get('length', 1.0)
            color = params.get('color', 'red')
            
            entity = sg.Cylinder(radius=radius, length=length, color=color)
            entity.T = pose
            
        elif primitive_type == "sphere":
            # Extract sphere parameters
            radius = params.get('radius', 0.5)
            color = params.get('color', 'green')
            
            entity = sg.Sphere(radius=radius, color=color)
            entity.T = pose

            
        else:
            raise ValueError(f"Unsupported primitive shape type: {primitive_type}")
        
        # Add to environment
        self.env.add(entity)
        
        # Store entity with an ID
        entity_id = str(len(self.entities))
        self.entities[entity_id] = entity
        
        return {
            "id": entity_id,
            "primitive_type": primitive_type,
            "params": params,
            "entity": entity
        }
    
    def get_mesh(self, obj: Any, transformed: bool = True) -> trimesh.Trimesh:
        """
        Extract mesh from the geometry.
        
        Args:
            obj: The geometry representation
            
        Returns:
            trimesh.Trimesh object
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj

        
        
        
        if hasattr(entity, 'to_dict'):
            obj_dict = entity.to_dict()
        else:   
            # TODO: add support for other object types
            raise ValueError("Cannot extract mesh from this object type")
        
        if obj_dict['stype'].lower() == 'cuboid':
            mesh = trimesh.creation.box(extents=obj_dict['scale'], transform=entity.T if transformed else None)
        elif obj_dict['stype'].lower() == 'cylinder':
            mesh = trimesh.creation.cylinder(radius=obj_dict['radius'], height=obj_dict['length'], transform=entity.T if transformed else None)
        elif obj_dict['stype'].lower() == 'sphere':
            mesh = trimesh.creation.icosphere(radius=obj_dict['radius'], transform=entity.T if transformed else None)
        else:
            raise ValueError("Cannot extract vertices from this object type")
        return mesh
       

    def get_vertices(self, obj: Any) -> np.ndarray:
        """
        Extract vertices from the geometry.
        
        Args:
            obj: The geometry representation
            
        Returns:
            Numpy array of vertices (Nx3) in world frame
        """
        mesh = self.get_mesh(obj)
        obb = mesh.bounding_box_oriented
        return np.array(obb.vertices)
        

    def get_faces(self, obj: Any) -> Optional[np.ndarray]:
        """
        Extract faces from the geometry, if available.
        
        Args:
            obj: The geometry representation
            
        Returns:
            Numpy array of face indices or None if not applicable
        """
        mesh = self.get_mesh(obj)
        return mesh.faces

    def to(self, transformation: Union[tuple, np.ndarray, sm.SE3], output_type: Optional[str] = 'pq') -> np.ndarray:
        """
        Convert input to the given output_type.
        
        Args:
            transformation: Can be SE3, transformation matrix, tuple of (position, quaternion), etc.
            output_type: Desired output format ('pq', 'transform', 'dual_quaternion', etc.)
            
        Returns:
            np.ndarray: Converted representation in the requested format
        """
        return convert_pose(transformation, output_type)

    def get_obb(self, obj: Union[Dict, Any]) -> Dict:
        """
        Calculate oriented bounding box for an object using trimesh.
        
        Args:
            obj: Object representation or direct entity
        
        Returns:
            Dictionary containing bounding box information
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj


        edge_indices = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face (clockwise)
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face (clockwise)
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges (bottom to top)
        ]
                    
        # Get object pose and size
        pose = self.get_pose(obj, 't')
        size = self.get_size(obj)
        center = pose[:3, 3]
        
        # Create corners in a specific order (bottom, up, clockwise)
        # Extract dimensions for clarity
        l, w, h = size  # length, width, height

        local_corners = np.array([
            [-l/2, -w/2, -h/2], 
            [ l/2, -w/2, -h/2],   
            [-l/2,  w/2, -h/2],   
            [ l/2,  w/2, -h/2],   
            [-l/2, -w/2,  h/2],    
            [ l/2, -w/2,  h/2],    
            [-l/2,  w/2,  h/2],     
            [ l/2,  w/2,  h/2]     
        ])
        
        # Transform corners to world coordinates
        corners = pt.transform(pose, np.hstack([local_corners, np.ones((8, 1))]))[:, :3]
        # corners = self.get_vertices(obj)
        # Calculate edges
        edges = np.zeros((len(edge_indices), 3))  # Shape: (12, 3)
        for i, (start, end) in enumerate(edge_indices):
            edges[i] = corners[end] - corners[start]
        
        # Calculate min/max bounds
        min_bounds = corners[0]  # bottom-back-left
        max_bounds = corners[7]  # top-front-right
        
        return {
            "vertices": corners,
            "edges": edges,
            "min_bounds": min_bounds,
            "max_bounds": max_bounds,
            "center": center,
            "transform": pose,
            "extents": size,
            'edge_indices': edge_indices
        }
    

    def get_bbox(self, obj: Union[Dict, Any]) -> Dict:
        """
        Calculate axis-aligned bounding box for an object.
        
        Args:
            obj: Object representation or direct entity
        
        Returns:
            Dictionary containing bounding box information
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj
        
        # Get vertices from the object
        vertices = self.get_vertices(obj)
        
        # Calculate AABB
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Generate corner vertices for AABB
        corner_template = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ])
        
        corners = np.zeros((8, 3))
        for i, corner in enumerate(corner_template):
            corners[i] = min_bounds + corner * (max_bounds - min_bounds)
        
        # Define edge indices
        edge_indices = [
            (0, 2), (4, 6), (0, 4), (2, 6),  # Bottom face
            (1, 3), (5, 7), (3, 7), (1, 5),  # Top face
            (0, 1), (2, 3), (4, 5), (6, 7)   # Connecting edges
        ]
        
        # Calculate edges
        edges = np.zeros((len(edge_indices), 3))
        for i, (start, end) in enumerate(edge_indices):
            edges[i] = corners[end] - corners[start]
        
        return {
            "vertices": corners,
            "edges": edges,
            "min_bounds": min_bounds,
            "max_bounds": max_bounds,
            'edge_indices': edge_indices
        }

    def get_pose(self, obj: Union[Dict, Any], output_type: Optional[str] = 'pq') -> np.ndarray:
        """
        Get the pose of the object.

        Args:
            obj: Object representation or direct entity
            output_type: Desired output format ('pq', 'transform', 'dual_quaternion', etc.)
        Returns:
            Pose of the object in the requested format
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj

        if hasattr(entity, 'T'):
            # object entity
            return self.to(entity.T, output_type)
        elif hasattr(entity, 'base'):
            # Robot entity - get base pose
            return self.to(entity.base, output_type)
        else:
            # Default to identity
            raise ValueError("Cannot get pose for this object type")

    def set_pose(self, obj: Union[Dict, Any], pose: Union[tuple, np.ndarray, sm.SE3]) -> None:
        """
        Set the pose of the object.

        Args:
            obj: Object representation or direct entity
            pose: Pose to set, can be SE3, (4, 4) transformation matrix, (7,) pq or (8,) dq
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj
        

        pose = self.to(pose, 'transform')
        entity.T = pose

        

    def get_size(self, obj: Union[Dict, Any]) -> np.ndarray:
        """
        Get the size of the bbox for the given object.

        Args:
            obj: Object representation or direct entity
        Returns:
            Size of the bbox (not the object), in (x, y, z) order
        """
        mesh = self.get_mesh(obj, transformed=False)
        return mesh.extents

    def transform(self, 
                  obj: Union[Dict, Any], 
                  transformation: Union[tuple, np.ndarray, sm.SE3], 
                  apply: bool = False,
                  output_type: Optional[str] = 't'
                  ) -> Union[np.ndarray, sm.SE3]:
        """
        Apply transformation to object.
        
        Args:
            obj: Object representation or direct entity
            transformation: Transformation to apply
            apply: Whether to apply the transformation to the object
            output_type: Desired output format

        Returns:
            Transformed pose of the object in the requested format
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj

        current_pose = self.get_pose(entity, output_type='transform')
        transformed_pose = current_pose * transformation

        if apply:
            self.set_pose(entity, transformed_pose)
            self.step_simulation(0.01)

        return self.to(transformed_pose, output_type)
    
    # Robot control methods
    

    def get_joint_positions(self, robot: Any) -> np.ndarray:
        """
        Get current joint positions of the robot.
        
        Args:
            robot: The robot representation
            
        Returns:
            Array of joint positions
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        if hasattr(entity, 'q'):
            return np.array(entity.q)
        else:
            raise ValueError("Object does not have joint positions")
        
    def get_joint_velocities(self, robot: Any) -> np.ndarray:
        """
        Get current joint velocities of the robot.
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        if hasattr(entity, 'qd'):
            return np.array(entity.qd)
        else:
            raise ValueError("Object does not have joint velocities")
    
    def set_joint_positions(self, robot: Any, positions: np.ndarray) -> None:
        """
        Set joint positions of the robot directly (without control).
        
        Args:
            robot: The robot representation
            positions: Array of joint positions
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        if hasattr(entity, 'q'):
            entity.q = positions
        else:
            raise ValueError("Object does not have joint positions")
    
    def control_joint_positions(self, robot: Any, positions: np.ndarray) -> None:
        """
        Control joint positions of the robot using PD controller.
        
        Args:
            robot: The robot representation
            positions: Array of joint positions (target)
        """
        # RTB doesn't have built-in control, just set directly
        self.set_joint_positions(robot, positions)
    
    def set_joint_velocities(self, robot: Any, velocities: np.ndarray) -> None:
        """
        Set joint velocities of the robot directly (without control).
        
        Args:
            robot: The robot representation
            velocities: Array of joint velocities
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        if hasattr(entity, 'qd'):
            entity.qd = velocities
        else:
            raise ValueError("Object does not have joint velocities")
    
    def control_joint_velocities(self, robot: Any, velocities: np.ndarray) -> None:
        """
        Control joint velocities of the robot using PD controller.
        
        Args:
            robot: The robot representation
            velocities: Array of joint velocities (target)
        """
        # RTB doesn't have built-in control, just set directly
        self.set_joint_velocities(robot, velocities)
    
    def get_dof(self, robot: Any) -> int:
        """
        Get the number of degrees of freedom of the robot.
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        return entity.n
    
    def get_joint_velocity_limits(self, robot: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint velocity limits of the robot.
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        if hasattr(entity, 'qdlim') and entity.qdlim is not None:
            return entity.qdlim[0, :], entity.qdlim[1, :]
        else:
            raise ValueError("Robot does not have joint velocity limits, set it using qdlim property")
    
    def get_joint_limits(self, robot: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint limits of the robot.
        
        Args:
            robot: The robot representation
            
        Returns:
            Tuple of (lower_limits, upper_limits)
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        if hasattr(entity, 'qlim'):
            limits = entity.qlim
            return limits[0, :], limits[1, :]  # lower, upper
        else:
            raise ValueError("Object does not have joint limits")
    
    def compute_jacobian(self, robot: Any, link: Any = None, base: bool = False) -> np.ndarray:
        """
        Compute the Jacobian matrix for the robot.
        
        Args:
            robot: The robot representation
            link: End-effector link (optional)
            
        Returns:
            Jacobian matrix
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        if base:
            return entity.jacob0(entity.q, end=link)
        else:
            return entity.jacobe(entity.q, end=link)
    
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
            End-effector pose in the requested format
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        if q is None:
            q = entity.q
            
        if hasattr(entity, 'fkine'):
            Te = entity.fkine(q, end=link)
            return self.to(Te, output_type)
        else:
            raise ValueError("Object does not support forward kinematics")
    
    def step_simulation(self, time: float) -> None:
        """
        Step the simulation forward by the specified time.
        
        Args:
            time: wallclock time to step the simulation
        """
        # Swift doesn't have explicit time stepping like Genesis
        # Instead, we can just call step() to update the visualization
        self.env.step(time)

 