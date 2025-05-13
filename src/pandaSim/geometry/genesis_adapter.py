"""
Genesis AI adapter implementation.

This module provides an adapter for the Genesis AI geometry backend.
"""
from typing import Any, Optional, Dict, Tuple, Union
import numpy as np
import genesis as gs
from pathlib import Path
from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)

class GenesisAdapter:
    """
    Adapter for Genesis AI geometry backend.
    
    Implements the GeometryAdapter protocol for Genesis AI.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Genesis AI adapter.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Configure viewer options
        viewer_options = self.config.get("viewer_options", {})
        self.viewer_options = gs.options.ViewerOptions(
            res=(viewer_options.get("width", 1280), viewer_options.get("height", 960)),
            camera_pos=viewer_options.get("camera_pos", (3.5, 0.0, 2.5)),
            camera_lookat=viewer_options.get("camera_lookat", (0.0, 0.0, 0.5)),
            camera_fov=viewer_options.get("fov", 40),
            max_FPS=viewer_options.get("max_fps", 60),
        )
        
        # Configure simulation options
        sim_options = self.config.get("sim_options", {})
        self.sim_options = gs.options.SimOptions(
            dt=sim_options.get("dt", 0.01),
        )
        
        # Configure visualization options
        vis_options = self.config.get("vis_options", {})
        self.vis_options = gs.options.VisOptions(
            show_world_frame=vis_options.get("show_world_frame", True),
            world_frame_size=vis_options.get("world_frame_size", 1.0),
            show_link_frame=vis_options.get("show_link_frame", False),
            show_cameras=vis_options.get("show_cameras", False),
            plane_reflection=vis_options.get("plane_reflection", True),
            ambient_light=vis_options.get("ambient_light", (0.1, 0.1, 0.1)),
        )
        
        # Create scene
        self.scene = gs.Scene(
            show_viewer=self.config.get("show_viewer", True),
            viewer_options=self.viewer_options,
            sim_options=self.sim_options,
            vis_options=self.vis_options,
            renderer=gs.renderers.Rasterizer(),
        )
        
        # Store loaded entities
        self.entities = {}
    
    @property
    def get_scene(self):
        """Get the underlying Genesis scene for direct access"""
        return self.scene
    
    def load(self, file: str) -> Any:
        """
        Load data from file using Genesis AI.
        
        Args:
            file_path: Path to the geometry file, robot model or object file
            
        Returns:
            Genesis AI representation of the geometry
        """

        
        # Determine the type of file and load accordingly
        file_path = Path(file)

        if file_path.suffix.lower() == '.xml':
            # Load MJCF file
            entity = self.scene.add_entity(
                gs.morphs.MJCF(file=str(file_path))
            )
        elif file_path.suffix.lower() == '.obj':
            # Load OBJ mesh file
            entity = self.scene.add_entity(
                gs.morphs.Mesh(file=str(file_path))
            )
        elif file_path.suffix.lower() == '.stl':
            # Load STL mesh file
            entity = self.scene.add_entity(
                gs.morphs.Mesh(file=str(file_path))
            )
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Store entity with an ID
        entity_id = str(len(self.entities))
        self.entities[entity_id] = entity
        
        return {
            "id": entity_id,
            "path": str(file),
            "entity": entity
        }
    
    def add_primitive(self, primitive_type: str, **params) -> Dict:
        """
        Add a primitive shape to the scene.
        
        Args:
            primitive_type: Type of primitive shape ("box", "cylinder", "sphere", "plane")
            **params: Shape-specific parameters including surface properties
                - pos, quat, euler: For shape position and orientation
                - size, lower, upper: For box shape
                - radius, height: For cylinder shape
                - radius: For sphere shape
                - surface: For material/color properties (e.g., gs.surfaces.Default(color=(r,g,b)))
        
        Returns:
            Dictionary containing the entity information
        """
        primitive_type = primitive_type.lower()
        
        # Extract surface parameter separately, don't pass it to the morph constructor
        surface = params.pop('surface', None)
        
        if primitive_type == "box":
            entity = self.scene.add_entity(
                gs.morphs.Box(**params),
                surface=surface
            )
            
        elif primitive_type == "cylinder":
            entity = self.scene.add_entity(
                gs.morphs.Cylinder(**params),
                surface=surface
            )
            
        elif primitive_type == "sphere":
            entity = self.scene.add_entity(
                gs.morphs.Sphere(**params),
                surface=surface
            )
            
        elif primitive_type == "plane":
            entity = self.scene.add_entity(
                gs.morphs.Plane(**params),
                surface=surface
            )
            
        else:
            raise ValueError(f"Unsupported primitive shape type: {primitive_type}")
        
        # Store entity with an ID
        entity_id = str(len(self.entities))
        self.entities[entity_id] = entity
        
        return {
            "id": entity_id,
            "primitive_type": primitive_type,
            "params": params,
            "entity": entity
        }


    
    def get_bbox(self, obj: Union[Dict, Any]) -> Dict:
        """
        Calculate axis-aligned bounding box for an object.
        
        Args:
            obj: Object representation or direct entity
        
        Returns:
            Dictionary containing bounding box information:
            - vertices: 8 corner points of the bounding box
            - edges: 12 edges connecting the vertices
            - min_bounds: Minimum coordinate values (x,y,z)
            - max_bounds: Maximum coordinate values (x,y,z)
        """
        # Handle both dict wrapper and direct entity
        entity = obj["entity"] if isinstance(obj, dict) else obj
        
        # Get mesh data from entity
        vertices = entity.get_verts().cpu().numpy()

        
        # Calculate min and max bounds
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Generate the 8 corners of the bounding box
        corners = []
        for x in [min_bounds[0], max_bounds[0]]:
            for y in [min_bounds[1], max_bounds[1]]:
                for z in [min_bounds[2], max_bounds[2]]:
                    corners.append([x, y, z])
        
        corners = np.array(corners)
        # Define the 12 edges as pairs of vertex indices
        edge_indices = [
            (0, 2), (4, 6), (0, 4), (2, 6),  # Bottom face
            (1, 3), (5, 7), (3, 7), (1, 5),  # Top face
            (0, 1), (2, 3), (4, 5), (6, 7)   # Connecting edges
        ]
        edges = [corners[j] - corners[i] for i, j in edge_indices]
        # edges = [edge / np.linalg.norm(edge) for edge in edges]
        edges = np.array(edges)
        

        return {
            "vertices": corners,
            "edges": edges,
            "min_bounds": min_bounds,
            "max_bounds": max_bounds
        }
    
    def transform(self, obj: Union[Dict, Any], transformation: np.ndarray) -> Dict:
        """
        Apply transformation to Genesis AI geometry.
        
        Args:
            obj: Genesis AI geometry representation
            transformation: 4x4 transformation matrix
            
        Returns:
            Transformed Genesis AI geometry
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj
        
        
        pq = pt.pq_from_transform(transformation)
        # Apply transformation to entity
        entity.set_qpos(
            pq
        )
    
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
        
        # Get joint positions using Genesis API
        return entity.get_dofs_position().cpu().numpy()
    
    def set_joint_positions(self, robot: Any, positions: np.ndarray) -> None:
        """
        Set joint positions of the robot directly (without control).
        
        Args:
            robot: The robot representation
            positions: Array of joint positions
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        # Set joint positions in Genesis
        entity.set_dofs_position(positions)
    
    def control_joint_positions(self, robot: Any, positions: np.ndarray) -> None:
        """
        Control joint positions of the robot using PD controller.
        
        Args:
            robot: The robot representation
            positions: Array of joint positions (target)
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        # Control joint positions using PD controller
        entity.control_dofs_position(positions)
    
    def set_joint_velocities(self, robot: Any, velocities: np.ndarray) -> None:
        """
        Set joint velocities of the robot directly (without control).
        
        Args:
            robot: The robot representation
            velocities: Array of joint velocities
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        # Set joint velocities in Genesis
        entity.set_dofs_velocity(velocities)
    
    def control_joint_velocities(self, robot: Any, velocities: np.ndarray) -> None:
        """
        Control joint velocities of the robot using PD controller.
        
        Args:
            robot: The robot representation
            velocities: Array of joint velocities (target)
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        # Control joint velocities using PD controller
        entity.control_dofs_velocity(velocities)
    
    def get_joint_limits(self, robot: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint limits of the robot.
        
        Args:
            robot: The robot representation
            
        Returns:
            Tuple of (lower_limits, upper_limits)
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        # Get joint limits from Genesis
        limits = entity.get_dofs_limit()
        lower_limits = limits[0].cpu().numpy()
        upper_limits = limits[1].cpu().numpy()
        
        return lower_limits, upper_limits
    
    def compute_jacobian(self, robot: Any, link: Any) -> np.ndarray:
        """
        Compute the Jacobian matrix for the robot.
        
        Args:
            robot: The robot representation
            link: End-effector link
            
        Returns:
            Jacobian matrix
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        
        jacobian = entity.get_jacobian(link=link).cpu().numpy()
        
        return jacobian
    
    def compute_forward_kinematics(self, 
                                   robot: Any, 
                                   link: Any = None, 
                                   q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute forward kinematics for the robot.
        
        Args:
            robot: The robot representation
            q: Joint positions (optional)
            link: End-effector link (optional)
            
        Returns:
            End-effector pose as 4x4 homogeneous transformation matrix
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        links_name = [link.name for link in entity.links]
        # Case 1: If both robot and joint positions are provided, use forward_kinematics
        if q is not None:
            # Use forward_kinematics to get position and orientation
            links_pos, links_quat = entity.forward_kinematics(q)
            
            # Convert to 4x4 transformation matrix
            # If we're interested in a specific link, use its index
            if link is not None:
                # If link is an integer index
                if isinstance(link, int):
                    pos = links_pos[link]
                    quat = links_quat[link]
                # If link is a string name, get its index
                elif isinstance(link, str):
                    link_idx = links_name.index(link)
                    pos = links_pos[link_idx]
                    quat = links_quat[link_idx]
                else:
                    # Default to the last link
                    pos = links_pos[-1]
                    quat = links_quat[-1]
            else:
                # Default to the last link (end-effector)
                pos = links_pos[-1]
                quat = links_quat[-1]
                
            # Convert position and quaternion to transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = pr.matrix_from_quaternion(quat.cpu().numpy())
            transform[:3, 3] = pos.cpu().numpy()
            
        # Case 2: If only link is provided, get position and quaternion directly
        elif link is not None:
            # Get position and quaternion
            if isinstance(link, int):
                link_name = links_name[link]
            elif isinstance(link, str):
                link_name = link
            elif 'Rigid' in str(type(link)):
                link_name = link.name
            else:
                link_name = links_name[-1]
            
            link = entity.get_link(link_name)
            
            # Convert to transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = pr.matrix_from_quaternion(link.get_quat().cpu().numpy())
            transform[:3, 3] = link.get_pos().cpu().numpy()
            
        # Default case: get current end-effector transform
        else:
            if 'RigidEntity' in str(type(entity)):
                # return all links transforms
                links_pos, links_quat = entity.forward_kinematics(
                    entity.get_dofs_position()
                    )
                transforms = ptr.transforms_from_pqs(
                    np.concatenate([links_pos, links_quat], axis=1),
                )
                return transforms
                
        return transform
    
    def step_simulation(self, time: float) -> None:
        """
        Step the simulation forward by dt seconds.
        
        Args:
            time: wallclock time to step the simulation
        """
        # Step the Genesis simulation
        dt = self.scene.sim_options.dt
        steps = int(time / dt)
        for _ in range(steps):
            self.scene.step()

