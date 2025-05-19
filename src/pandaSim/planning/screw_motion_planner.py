"""
Screw motion planner implementation.

This module implements planning for object reorientation using screw motion.
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import numpy.linalg as LA
import torch
from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)

from pandaSim.planning.protocols import PlannerStrategy
from pandaSim.geometry.protocols import GeometryAdapter
from pandaSim.geometry.utils import convert_pose


class ScrewMotionPlanner(PlannerStrategy):
    """
    Planner that generates screw motion trajectories for object reorientation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the planner with configuration options."""
        self.config = config or {}
        self.ground_height_tolerance = self.config.get("ground_height_tolerance", 0.01)
        self.gripper_max_size = self.config.get("gripper_max_size", 0.08)  # 8cm
        
    def find_ground_edges(self, bbox: Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Find edges that lie on the ground plane.
        
        Args:
            bbox: Bounding box from adapter.get_bbox()
            
        Returns:
            pairs_per_env: List of edge pairs
            edges_per_env: List of edge vectors
        """
        vertices = bbox["vertices"]
        edges = bbox["edges"]
        # fixed list of the 12 vertex‐pairs
        edge_pairs = np.array(bbox['edge_indices'])                # (12, 2)
        
        # --- handle single‐env case by adding a batch dimension ---
        if vertices.ndim   == 2:  # (8,3) → (1,8,3)
            vertices = vertices[None, ...]
        if edges.ndim      == 2:  # (12,3) →  (1,12,3)
            edges = edges[None, ...]


        # Define ground plane height with tolerance (min z coordinate of the vertices)
        ground_height = vertices[..., 2].min(axis=-1)             # (n_envs,)
        # Find vertices with z coordinate close to ground height
        z_coords = vertices[..., 2]                      # (n_envs, 8)
        ground_height_expanded = ground_height[:, None]  # (n_envs, 1)
        mask = np.abs(z_coords - ground_height_expanded) < self.ground_height_tolerance
                                                        # (n_envs, 8)



        # which edges have both endpoints on the ground?
        edge_on_ground = mask[:, edge_pairs].all(axis=2)  # (n_envs, 12)

        pairs_per_env = [edge_pairs[row]    for row in edge_on_ground]
        edges_per_env = [edges[env, row]    for env, row in enumerate(edge_on_ground)]

        return pairs_per_env, edges_per_env

    
        

        
    def screw_from_bbox(self, 
                        bbox: Dict, 
                        base_pos: np.ndarray = None,
                        prefer_closest: bool = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the optimal edge and vertex for screw motion.
        
        Args:
            bbox: Bounding box from adapter.get_bbox(obj)
            base_pos: Position of the robot base, or EE. Screw axes will be sorted based on distance from this point.
            prefer_closest: If True, select the screw axis closest to the base_pos, otherwise select the farthest
        Returns:
            qs: List of points on screw axis, (n_envs, n_valid_qs, 3)
            s_axes: List of unit vectors along screw axis, (n_envs, n_valid_qs, 3)
        """
        vertices = bbox["vertices"]
        
        if vertices.ndim   == 2:  # (8,3) → (1,8,3)
            vertices = vertices[None, ...]
        
        # 1. Find edges on the ground
        pairs_per_env, edges_per_env = self.find_ground_edges(bbox)

        if not pairs_per_env:
            raise ValueError("No ground edges found for screw motion")
        
        # 2. Filter edges by length (must be smaller than gripper size)
        edge_mask = [np.linalg.norm(edge, axis=1) <= self.gripper_max_size for edge in edges_per_env]
        
        
        qs = []
        s_axes = []
        n_envs = len(pairs_per_env)
        
        for env_idx, (pairs, edges, mask) in enumerate(zip(pairs_per_env, edges_per_env, edge_mask)):
            valid_edges = edges[mask]
            valid_pairs = pairs[mask]

            if len(valid_edges) == 0:
                # If no valid edges, use any ground edge
                valid_edges = edges
                valid_pairs = pairs
            
            if len(valid_pairs) == 0:
                qs.append([])
                s_axes.append([])
                continue

            # Calculate distances from robot base to edge midpoints
            midpoints = []
            for pair in valid_pairs:
                v1 = vertices[env_idx, pair[0]]
                v2 = vertices[env_idx, pair[1]]
                midpoint = (v1 + v2) / 2
                midpoints.append(midpoint)

            
            midpoints = np.array(midpoints)
            if base_pos is None:
                base_pos = np.tile([0, 0, 0], (n_envs, 1))

            distances = np.linalg.norm(midpoints - base_pos[env_idx], axis=1)
            # Sort edges based on distance from base
            sorted_indices = np.argsort(distances)
            sorted_edges = valid_edges[sorted_indices]
            sorted_edges = sorted_edges / np.linalg.norm(sorted_edges, axis=1, keepdims=True)
            sorted_midpoints = midpoints[sorted_indices]
            qs.append(sorted_midpoints)
            s_axes.append(sorted_edges)

        if prefer_closest is not None:
            if prefer_closest:
                qs = [q[0, :] for q in qs]
                s_axes = [s_axis[0, :] for s_axis in s_axes]
            else:
                qs = [q[-1, :] for q in qs]
                s_axes = [s_axis[-1, :] for s_axis in s_axes]

        return qs, s_axes

    
    def time_scaling (self,
                      steps: int = 300,
                      method: str = 'quintic') -> np.ndarray:
        """
        Generate time scaling for trajectory.
        
        Args:
            steps: Number of waypoints to generate
            method: Time scaling method (quintic, cubic, linear, etc.)

        Returns:
            tau: Time scaling factor for each waypoint
        """
        s = np.linspace(0, 1, steps)
        if method.lower().startswith('l'):
            return s
        elif method.lower().startswith('c'):
            return 3*s**2 - 2*s**3
        elif method.lower().startswith('q'):
            return 10*s**3 - 15*s**4 + 6*s**5
        else:
            raise ValueError(f"Invalid time scaling method: {method}, choose from q, c, l")

    def generate_screw_trajectory(self, 
                                  initial_pose: tuple | torch.Tensor | np.ndarray,
                                  q: np.ndarray, 
                                  s_axis: np.ndarray, 
                                  theta: float = np.pi/2,
                                  h: float = 0.0, 
                                  steps: Optional[int] = 300,
                                  tau: Optional[np.ndarray] = None,
                                  time_scaling: Optional[str] = 'q',
                                  output_type: Optional[str] = 'dq'
                                  ) -> List[np.ndarray]:
        """
        Generate trajectory waypoints using screw motion.
        
        Args:
            initial_pose: array-like, shape (4, 4), (7,), (8,)
                Initial pose of the robot EE, or object(T, dual quaternion, or pq)
            s_axis: array-like, shape (3,)
                Unit vector along screw axis
            q: array-like, shape (3,)
                Point on screw axis
            theta: float
                Total rotation angle (default: 90 degrees)
            h: float
                Screw pitch (default: 0 = pure rotation)
            steps: int
                Number of waypoints to generate
            tau: array-like, shape (steps,)
                Time scaling factor for each waypoint
            time_scaling: str
                Time scaling method (quintic, cubic, linear, etc.)
            output_type: str
                Output type (dq, pq, T)
        Returns:
            List of dual quaternions representing the trajectory
        """
        
        if tau is None:
            tau = self.time_scaling(steps, time_scaling)
        
        initial_pose = convert_pose(initial_pose, 'dq')
        screw_dq = pt.dual_quaternion_from_screw_parameters(q=q, s_axis=s_axis, h=h, theta=theta)
        goal_pose = pt.concatenate_dual_quaternions(screw_dq, initial_pose)
        
        traj = [pt.dual_quaternion_sclerp(initial_pose, goal_pose, t) for t in tau]

        if output_type.lower().startswith('t'):
            return ptr.transforms_from_dual_quaternions(traj)

        elif output_type.lower().startswith('p'):
            return ptr.pqs_from_dual_quaternions(traj)

        elif output_type.lower().startswith('d'):
            return np.array(traj)

        else:
            raise ValueError(f"Invalid output type: {output_type}, must be T, pq, dq")

    def compute_grasp(self, 
                           obj: Any,
                           adapter: GeometryAdapter,
                           prefer_closer_grasp: bool = True,
                           base_pos: Optional[np.ndarray] = None,
                           grasp_height: Optional[str] = 'center',
                           gripper_depth: float = 0.03,
                           gripper_offset: Optional[np.ndarray] = None,
                           output_type: Optional[str] = 't') -> np.ndarray:
        """
        Compute optimal grasp point (middle point of robot fingers) for an object.
        
        Args:
            obj: The object representation
            adapter: Geometry adapter
            prefer_closer_grasp: If True, select the grasp point closer to the base_pos, otherwise select the farthest
            base_pos: Position of the robot base, or EE. Grasp points will be sorted based on distance from this point.
            grasp_height: 'center', 'top', 'bottom' - vertical position for grasping
            gripper_depth: Depth of the gripper fingers in meters
            gripper_offset: Offset transformation for the gripper fingers (grasp pose is at the center of the gripper fingers, 
            if the link is offset from the center of the gripper fingers, this offset is applied, if provided, the grasp_point will be grasp_pose)
        Returns:
            Grasp point as numpy array with shape (n_envs, 3)
        """
        # Get object pose and bounding box
        obb = adapter.get_obb(obj)
        wTobj = adapter.get_pose(obj, 't')
        wRobj = wTobj[..., :3, :3]
        objRw = np.einsum('...ij->...ji', wRobj)
        object_center = wTobj[..., :3, 3]
        object_size_local = adapter.get_size(obj)
        vertices = obb["vertices"]

        max_height = vertices[..., 2].max(axis=-1)
        min_height = vertices[..., 2].min(axis=-1)

        # Get screw axis and point on axis
        qs, s_axes = self.screw_from_bbox(
            bbox=obb, 
            base_pos=base_pos, 
            prefer_closest=not prefer_closer_grasp
        )




        # Vector from screw axis to object center
        to_center = object_center - qs
        
        # Project this vector onto the XY plane for horizontal approach
        direction = to_center.copy()
        direction[..., 2] = 0 #projection onto XY plane


        # Find direction normal for each env
        direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction_unit = direction / np.where(direction_norm > 1e-6, direction_norm, 1e-6)

        direction_unit_obj = np.einsum('...ij,...j->...i', objRw, direction_unit)
        direction_unit_obj = direction_unit_obj / np.linalg.norm(direction_unit_obj, axis=-1, keepdims=True)

        offset_obj = direction_unit_obj * object_size_local / 2 - gripper_depth * direction_unit_obj
        offset_w = (wTobj[..., :3, :3] @ offset_obj[..., :, None])[..., 0] + wTobj[..., :3, 3]

        grasp_point_xy = offset_w

        # Compute grasp height based on preference
        if grasp_height == 'center':
            z_height = object_center[..., 2]
        elif grasp_height == 'top':
            z_height = max_height - gripper_depth
        elif grasp_height == 'bottom':
            z_height = min_height + gripper_depth
        else:
            # Default to center if invalid option provided
            z_height = object_center[:, 2]
            
        # Final grasp point
        grasp_point = np.column_stack([grasp_point_xy[:, 0], grasp_point_xy[:, 1], z_height])

        if gripper_offset is not None:
            grasp_T_gripper = convert_pose(gripper_offset, 't')
            
            w_T_grasp = wTobj.copy()
            w_T_grasp[..., :3, 3] = grasp_point
            
            if w_T_grasp.ndim == 3:
                w_T_gripper = ptr.concat_one_to_many(grasp_T_gripper, w_T_grasp)
            else:
                w_T_gripper = pt.concat(grasp_T_gripper, w_T_grasp)

            return convert_pose(w_T_gripper, output_type).squeeze(), qs, s_axes
        
        return grasp_point.squeeze(), qs, s_axes
        

    def plan(self, 
             robot: Any, 
             link: Any,
             object: Any, 
             adapter: GeometryAdapter,
             initial_pose: Optional[np.ndarray] = None,
             prefer_closer_grasp: bool = True,
             base_pos: Optional[np.ndarray | str] = None,
             qs: Optional[List[np.ndarray]] = None,
             s_axes: Optional[List[np.ndarray]] = None,
             theta: float = np.pi/2,
             h: float = 0.0,
             steps: int = 300,
             time_scaling: str = 'q',
             output_type: str = 'pq') -> List[np.ndarray]:
        """
        Plan trajectory for object reorientation using screw motion.
        
        Args:
            robot: The robot representation
            link: The link representation
            object: The object representation
            adapter: Geometry adapter for accessing geometry
            initial_pose: Initial pose of to apply screw motion to
                If None, the current pose of the robot link will be used
            prefer_closest: If True, select the screw axis closest to base_pos, otherwise select the farthest
            base_pos: Position to measure distance from when selecting screw axis (defaults to robot base)
            theta: Rotation angle in radians (default: π/2)
            h: Screw pitch for translation along axis (default: 0.0)
            steps: Number of waypoints in trajectory (default: 100)
            time_scaling: Time scaling method ('q'=quintic, 'c'=cubic, 'l'=linear)
            output_type: Output format ('pq', 'dq', or 't' for transformation matrices)
            grasp_pose: Pre-computed grasp pose from compute_grasp
            qs: Pre-computed points on screw axes from compute_grasp
            s_axes: Pre-computed screw axes from compute_grasp
            
        Returns:
            List of poses representing the trajectory, with length n_envs and each shape (steps, output_type dims)
        """
        # Get robot base position if not provided
        if base_pos is None:
            # Use robot base position - assuming first link is base
            if hasattr(robot, "get_pos"):
                base_pos = robot.get_pos().cpu().numpy()
        elif isinstance(base_pos, str) and base_pos.lower().startswith("e"):
            base_pos = adapter.forward_kinematics(robot, link, output_type='pq')[..., :3]
        else:
            base_pos = np.array(base_pos)
        
        # Get current robot end-effector pose if initial_pose not provided
        if initial_pose is None:
            initial_pose = adapter.forward_kinematics(robot, link, output_type='dq')
        
        # If grasp info not provided, compute it
        if qs is None or s_axes is None:
            _, qs, s_axes = self.compute_grasp(
                obj=object,
                adapter=adapter,
                prefer_closer_grasp=prefer_closer_grasp,
                base_pos=base_pos
            )
        
        # Generate trajectories for each environment
        all_trajectories = []
        
        # Determine if we're dealing with single or multiple environments
        is_multi_env = adapter.scene.n_envs > 0
        n_envs = adapter.scene.n_envs if is_multi_env else 1
        # Ensure initial_pose is properly formatted for iteration
        if not isinstance(initial_pose, list) and n_envs > 1:
            # If single pose provided for multiple environments, duplicate it
            if hasattr(initial_pose, "ndim") and initial_pose.ndim == 1:
                initial_pose = [initial_pose] * n_envs
        for env_idx in range(n_envs):
            # Get data for this environment
            env_q = qs[env_idx] if is_multi_env else np.array(qs).squeeze()
            env_s_axis = s_axes[env_idx] if is_multi_env else np.array(s_axes).squeeze()
            
            if len(env_q) == 0 if isinstance(env_q, list) else False:
                raise ValueError(f"No valid screw axes found for environment {env_idx}")
            
            # Get the current pose for this environment
            if initial_pose is not None:
                if isinstance(initial_pose, list) or (hasattr(initial_pose, "ndim") and initial_pose.ndim > 1 and len(initial_pose) > 1):
                    # Multi-environment case
                    current_pose = initial_pose[env_idx]
                else:
                    # Single environment case
                    current_pose = initial_pose
            
            # Generate trajectory
            traj = self.generate_screw_trajectory(
                initial_pose=current_pose,
                q=env_q,
                s_axis=env_s_axis,
                theta=theta,
                h=h,
                steps=steps,
                time_scaling=time_scaling,
                output_type=output_type
            )
            
            all_trajectories.append(traj)
        
        # Convert to numpy array
        result = np.array(all_trajectories)
        
        return result.squeeze()