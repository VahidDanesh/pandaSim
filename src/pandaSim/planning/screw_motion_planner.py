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
        min_bounds = bbox["min_bounds"]

        # --- handle single‐env case by adding a batch dimension ---
        if vertices.ndim   == 2:  # (8,3) → (1,8,3)
            vertices = vertices[None, ...]
        if edges.ndim      == 2:  # (12,3) →  (1,12,3)
            edges = edges[None, ...]
        if min_bounds.ndim == 1:  # (3,) → (1,3)
            min_bounds = min_bounds[None, ...]

        # Define ground plane height with tolerance
        ground_height = min_bounds[..., 2]               # (n_envs,)
        
        # Find vertices with z coordinate close to ground height
        z_coords = vertices[..., 2]                      # (n_envs, 8)
        ground_height_expanded = ground_height[:, None]  # (n_envs, 1)
        mask = np.abs(z_coords - ground_height_expanded) < self.ground_height_tolerance
                                                        # (n_envs, 8)

        # fixed list of the 12 vertex‐pairs
        edge_pairs = np.array([
            (0,2),(4,6),(0,4),(2,6),   # bottom face
            (1,3),(5,7),(3,7),(1,5),   # top face
            (0,1),(2,3),(4,5),(6,7)    # side connectors
        ], dtype=int)                  # (12, 2)

        # which edges have both endpoints on the ground?
        edge_on_ground = mask[:, edge_pairs].all(axis=2)  # (n_envs, 12)

        pairs_per_env = [edge_pairs[row]    for row in edge_on_ground]
        edges_per_env = [edges[env, row]    for env, row in enumerate(edge_on_ground)]

        return pairs_per_env, edges_per_env

    
        

        
    def screw_from_bbox(self, 
                        bbox: Dict, 
                        base_pos: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the optimal edge and vertex for screw motion.
        
        Args:
            bbox: Bounding box from adapter.get_bbox(obj)
            base_pos: Position of the robot base, or EE. Screw axes will be sorted based on distance from this point.
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
        
        for env_idx, (pairs, edges, mask) in enumerate(zip(pairs_per_env, edges_per_env, edge_mask)):
            valid_edges = edges[mask]
            valid_pairs = pairs[mask]

            if len(valid_edges) == 0:
                # If no valid edges, use any ground edge
                valid_edges = edges
                valid_pairs = pairs
            
            # Calculate distances from robot base to edge midpoints
            midpoints = []
            for pair in valid_pairs:
                v1 = vertices[env_idx, pair[0]]
                v2 = vertices[env_idx, pair[1]]
                midpoint = (v1 + v2) / 2
                midpoints.append(midpoint)
            
            midpoints = np.array(midpoints)

            if base_pos is None:
                base_pos = np.array([0, 0, 0])
                 
            distances = np.linalg.norm(midpoints - base_pos[env_idx], axis=1)
            # Sort edges based on distance from base
            sorted_indices = np.argsort(distances)
            sorted_edges = valid_edges[sorted_indices]
            sorted_edges = sorted_edges / np.linalg.norm(sorted_edges, axis=1, keepdims=True)
            sorted_midpoints = midpoints[sorted_indices]
            
            qs.append(sorted_midpoints)
            s_axes.append(sorted_edges)

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

        
        
    
    def plan(self, 
             robot: Any, 
             link: Any,
             object: Any, 
             adapter: GeometryAdapter,
             initial_pose: Optional[np.ndarray] = None,
             prefer_closest: bool = True,
             base_pos: Optional[np.ndarray | str] = None,
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
        
        # Get current robot end-effector pose
        ee_pose_dq = adapter.forward_kinematics(robot, link, output_type='dq')
        
        # Get object bounding box
        bbox = adapter.get_bbox(object)
        
        # Get candidate screw axes and points
        
        qs_per_env, s_axes_per_env = self.screw_from_bbox(bbox, base_pos)
        
        # Generate trajectories for each environment
        all_trajectories = []
        
        # Determine number of environments
        n_envs = len(qs_per_env)
        
        for env_idx in range(n_envs):
            # Get candidates for this environment
            qs = qs_per_env[env_idx]
            s_axes = s_axes_per_env[env_idx]
            
            if len(qs) == 0:
                raise ValueError(f"No valid screw axes found for environment {env_idx}")
            
            # Select appropriate screw axis (first one is closest, last one is farthest)
            q_idx = 0 if prefer_closest else -1
            q = qs[q_idx]
            s_axis = s_axes[q_idx]
            
            if initial_pose is not None:
                current_pose = initial_pose[env_idx] 
            else:
                # Get the current pose for this environment
                if isinstance(ee_pose_dq, np.ndarray) and ee_pose_dq.ndim > 1 and len(ee_pose_dq) > 1:
                    # Multi-environment case
                    current_pose = ee_pose_dq[env_idx]
                else:
                # Single environment case
                    current_pose = ee_pose_dq
            
            # Generate trajectory
            traj = self.generate_screw_trajectory(
                initial_pose=current_pose,
                q=q,
                s_axis=s_axis,
                theta=theta,
                h=h,
                steps=steps,
                time_scaling=time_scaling,
                output_type=output_type
            )
            
            all_trajectories.append(traj)
        
        # Convert to numpy array
        result = np.array(all_trajectories)
        
        
        return result