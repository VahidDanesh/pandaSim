"""
Genesis AI adapter implementation.

This module provides an adapter for the Genesis AI geometry backend.
"""
from typing import Any, Optional, Dict, Tuple, Union
import numpy as np
import genesis as gs
import torch
import trimesh
from pathlib import Path
from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)

from pandaSim.geometry.utils import convert_pose

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
            substeps = sim_options.get("substeps", 8),
            
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
            rigid_options=gs.options.RigidOptions(
                enable_multi_contact=self.config.get('enable_multi_contact', True)),
            renderer=gs.renderers.Rasterizer(),
            show_FPS=self.config.get('show_FPS', False),
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

    def to(self, transformation: tuple | torch.Tensor | np.ndarray, output_type: Optional[str] = 'pq') -> np.ndarray:
        """
        Convert input to the given output_type.
        
        Args:
            input: Can be one of the following:
                - tuple of (position(s), quaternion(s))
                - pq(s) format (x, y, z, qw, qx, qy, qz)
                - dual quaternion(s) (qw, qx, qy, qz, tx, ty, tz)
                - transformation matrix(ices) 
            output_type: Desired output format ('pq', 'transform', 'dual_quaternion', etc.)
            
        Returns:
            np.ndarray: Converted representation in the requested format
        """
        return convert_pose(transformation, output_type)

    def get_obb(self, obj: Union[Dict, Any]) -> Dict:
        """
        Calculate oriented bounding box (OBB) for an object using trimesh.
        
        Args:
            obj: Object representation or direct entity
        
        Returns:
            Dictionary containing bounding box information:
            - vertices: 8 corner points of the bounding box
            - edges: 12 edges connecting the vertices
            - min_bounds: Minimum coordinate values in OBB space
            - max_bounds: Maximum coordinate values in OBB space
            - center: Center point of the OBB
            - axes: Principal axes of the OBB (3x3 rotation matrix)
            - extents: Full lengths of the OBB along its principal axes
        """
        # Handle both dict wrapper and direct entity
        entity = obj["entity"] if isinstance(obj, dict) else obj
        
        # Get mesh data from entity
        vertices = entity.get_verts().cpu().numpy()
        
        # Handle both single and multi-environment cases
        n_envs = self.scene.n_envs
        edge_indices = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face (clockwise)
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face (clockwise)
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges (bottom to top)
        ]



            
        if n_envs == 0:

            
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
            rotation = pose[:3, :3]
            corners = np.array([rotation @ corner + center for corner in local_corners])
            
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
        
        else:
            # Multi-environment case
            result_corners = []
            result_centers = []
            result_edges = []
            result_transforms = []
            result_min_bounds = []
            result_max_bounds = []
            result_extents = []
            
            for i in range(n_envs):
                # Get object pose and size for this environment
                pose = self.get_pose(obj, 't', env_idx=i)
                size = self.get_size(obj)
                center = pose[:3, 3]
                
                # Extract dimensions for clarity
                l, w, h = size[i, :] # length, width, height

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
                rotation = pose[:3, :3]
                corners = np.array([rotation @ corner + center for corner in local_corners])
                
                result_corners.append(corners)
                result_centers.append(center)
                result_transforms.append(pose)
                result_extents.append(size)
                
                # Calculate min/max bounds
                min_bounds = corners[0]  # bottom-back-left
                max_bounds = corners[7]  # top-front-right
                
                result_min_bounds.append(min_bounds)
                result_max_bounds.append(max_bounds)
                
                # Calculate edges
                edges = np.zeros((len(edge_indices), 3))
                for j, (start, end) in enumerate(edge_indices):
                    edges[j] = corners[end] - corners[start]
                    
                result_edges.append(edges)
                
            # Stack all results
            corners = np.stack(result_corners)
            centers = np.stack(result_centers)
            edges = np.stack(result_edges)
            transforms = np.stack(result_transforms)
            extents = np.stack(result_extents)
            min_bounds = np.stack(result_min_bounds)
            max_bounds = np.stack(result_max_bounds)
            
            return {
                "vertices": corners,
                "edges": edges,
                "min_bounds": min_bounds,
                "max_bounds": max_bounds,
                "center": centers,
                "transforms": transforms,
                "extents": extents,
                'edge_indices': edge_indices
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
        
        # Define edge indices once - these don't change
        edge_indices = [
            (0, 2), (4, 6), (0, 4), (2, 6),  # Bottom face
            (1, 3), (5, 7), (3, 7), (1, 5),  # Top face
            (0, 1), (2, 3), (4, 5), (6, 7)   # Connecting edges
        ]
        
        # Create corner pattern for the box (which corners get min vs max)
        corner_template = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ])
        
        # Handle both single and multi-environment cases uniformly
        if len(vertices.shape) == 3:  # Multiple environments: (n_envs, n_vertices, 3)
            # Compute min and max bounds along the vertices dimension
            min_bounds = np.min(vertices, axis=1)  # Shape: (n_envs, 3)
            max_bounds = np.max(vertices, axis=1)  # Shape: (n_envs, 3)
            
            # Expand dimensions for broadcasting with corner template
            min_expanded = np.expand_dims(min_bounds, axis=1)  # Shape: (n_envs, 1, 3)
            max_expanded = np.expand_dims(max_bounds, axis=1)  # Shape: (n_envs, 1, 3)
            
            # Generate corners by interpolating between min and max bounds
            corners = min_expanded + corner_template * (max_expanded - min_expanded)  # Shape: (n_envs, 8, 3)
            
            # Calculate edges
            edges = np.zeros((vertices.shape[0], len(edge_indices), 3))  # Shape: (n_envs, 12, 3)
            for i, (start, end) in enumerate(edge_indices):
                edges[:, i] = corners[:, end] - corners[:, start]
        
        else:  # Single environment: (n_vertices, 3)
            # Compute min and max bounds
            min_bounds = np.min(vertices, axis=0)  # Shape: (3,)
            max_bounds = np.max(vertices, axis=0)  # Shape: (3,)
            
            # Generate corners
            corners = np.zeros((8, 3))
            for i, corner in enumerate(corner_template):
                corners[i] = min_bounds + corner * (max_bounds - min_bounds)
            
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

        pos, quat = entity.get_pos(), entity.get_quat()
        
        return self.to((pos, quat), output_type)
        


    def get_size(self, obj: Union[Dict, Any]) -> np.ndarray:
        """
        Get the size of the bbox for the given object, represented in object's coordinate frame.

        Args:
            obj: Object representation or direct entity
        Returns:
            Size of the bbox, in (x, y, z) order of it's own coordinate frame. shape is (3,) 
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj
        # Create a trimesh mesh from vertices
        vertices = entity.get_verts().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices)
        
        # Get the oriented bounding box
        obb = mesh.bounding_box_oriented
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh, angle_digits=3)
        size = extents
        
        # bbox = self.get_bbox(obj)
        # size_world = bbox['max_bounds'] - bbox['min_bounds'] # in world frame
        # wTb = self.get_pose(entity, 't')
        # wRb = wTb[..., :3, :3]
        # bRw = np.linalg.inv(wRb)
        # size = np.einsum('nij,nj->ni', bRw, size_world)
        
        return np.abs(np.round(size, 3)) # round to compensate for bbox inaccuracy
        
    
    def set_pose(self, obj: Union[Dict, Any], pose: tuple | np.ndarray | torch.Tensor, envs_idx=None) -> None:
        """
        Set the pose of the object.

        Args:
            obj: Object representation or direct entity
            pose: Pose to set, can be (4, 4) transformation matrix, (7,) pq or (8,) dq
            envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.


        
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj
        
        pq = self.to(pose, 'pq')

        pos = pq[..., :3]
        quat = pq[..., 3:]
        
        entity.set_pos(pos, envs_idx=envs_idx)
        entity.set_quat(quat, envs_idx=envs_idx)
        
            
        

    def transform(self, 
                  obj: Union[Dict, Any], 
                  transformation: tuple | np.ndarray | torch.Tensor, 
                  apply: bool = False,
                  output_type: Optional[str] = 't',
                  envs_idx=None
                  ) -> Dict:
        """
        Apply transformation to object.
        
        Args:
            obj: Object representation or direct entity
            transformation: Transformation to apply, can be (4, 4) transformation matrix, (7,) pq or (8,) dq or tuple of (position, quaternion).
                            Can be shape (n_envs, ...) to apply different transformations to each environment, or a single transformation to apply to all.
            apply: Whether to apply the transformation to the object
            output_type: Desired output format ('t', 'pq', 'dq', etc.)
            envs_idx: None | array_like, optional
                      The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns:
            Transformed pose of the object in the requested format
        """
        entity = obj["entity"] if isinstance(obj, dict) else obj

        # Convert transformation to transformation matrix format
        transformation = self.to(transformation, 't')
        
        # Get object's current pose
        object_pose = self.get_pose(entity, output_type='t')
        
        # Handle single transformation applied to multiple environments
        if len(transformation.shape) == 2 and len(object_pose.shape) > 2:
            # Expand single transformation to match number of environments
            transformation = np.tile(transformation, (object_pose.shape[0], 1, 1))
        
        # Apply transformation only to specified environments if envs_idx is provided
        if envs_idx is not None:
            # Apply transformation only to specified environments
            selected_object_pose = object_pose[envs_idx]
            transformation = transformation[envs_idx] if transformation.ndim == 3 else transformation
            transformed_pose = pt.concat(transformation, selected_object_pose)

        else:
            # Apply transformation to all environments
            transformed_pose = ptr.concat_many_to_many(transformation, object_pose)

        if apply:
            self.set_pose(entity, transformed_pose, envs_idx=envs_idx)

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
    
    def forward_kinematics(self, 
                            robot: Any, 
                            link: Any = None, 
                            q: Optional[np.ndarray] = None,
                            output_type: Optional[str] = 'dq') -> np.ndarray:
        """
        Compute forward kinematics for the robot.
        
        Args:
            robot: The robot representation
            q: Joint positions (optional)
            link: End-effector link (optional)
            output_type: Type of output ('t', 'dq', 'pq')
            
        Returns:
            End-effector or all links pose as 4x4 homogeneous transformation matrix or dual quaternion or pq
        """
        entity = robot["entity"] if isinstance(robot, dict) else robot
        links_name = [link.name for link in entity.links]
        output_type = output_type.lower()

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
            pos = link.get_pos()
            quat = link.get_quat()
            
            

            
        # Default case: get current end-effector transform
        else:
            if 'RigidEntity' in str(type(entity)):
                # return all links transforms
                pos, quat = entity.forward_kinematics(
                    entity.get_dofs_position()
                    )
        
        pq = np.hstack((pos.cpu().numpy(), quat.cpu().numpy()))

        if output_type.startswith('t'):
            return ptr.transforms_from_pqs(pq)
        elif output_type.startswith('dq'):
            return ptr.dual_quaternions_from_pqs(pq)
        else:
            return pq
    
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

