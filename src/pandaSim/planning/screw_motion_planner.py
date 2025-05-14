"""
Screw motion planner implementation.

This module implements planning for object reorientation using screw motion.
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import numpy.linalg as LA

from pytransform3d import (
    transformations as pt,
    rotations as pr,
    batch_rotations as pb,
    trajectories as ptr,
    plot_utils as ppu
)

from pandaSim.planning.protocols import PlannerStrategy
from pandaSim.geometry.protocols import GeometryAdapter


class ScrewMotionPlanner(PlannerStrategy):
    """
    Planner that generates screw motion trajectories for object reorientation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the planner with configuration options."""
        self.config = config or {}
        self.ground_height_tolerance = self.config.get("ground_height_tolerance", 0.01)
        self.gripper_max_size = self.config.get("gripper_max_size", 0.08)  # 8cm
        
    def find_ground_edges(self, bbox: Dict) -> List[Tuple[int, np.ndarray]]:
        """
        Find edges that lie on the ground plane.
        
        Args:
            bbox: Bounding box from adapter.get_bbox()
            
        Returns:
            List of (edge_index, edge_vector) tuples for edges on ground
        """
        vertices = bbox["vertices"]
        edges = bbox["edges"]
        min_bounds = bbox["min_bounds"]
        
        # Define ground plane height with some tolerance
        ground_height = min_bounds[2]
        
        # Find vertices on the ground (z coordinate close to min_z)
        ground_vertices_indices = [
            i for i, v in enumerate(vertices) 
            if abs(v[2] - ground_height) < self.ground_height_tolerance
        ]

        # Define edge connections between vertices (from the AABB definition)
        # Bottom face edges: (0,2), (4,6), (0,4), (2,6)
        edge_connections = [(0, 2), (4, 6), (0, 4), (2, 6)]
        
        # Find edges where both vertices are on the ground
        ground_edges = []
        for i, (v1_idx, v2_idx) in enumerate(edge_connections):
            if v1_idx in ground_vertices_indices and v2_idx in ground_vertices_indices:
                ground_edges.append(((v1_idx, v2_idx), edges[i]))
        
        return ground_edges
        
    def screw_from_bbox(self, 
                         bbox: Dict, 
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the optimal edge and vertex for screw motion.
        
        Args:
            bbox: Bounding box from adapter.get_bbox()
            
        Returns:
            qs: List of points on screw axis
            s_axes: List of unit vectors along screw axis
        """
        vertices = bbox["vertices"]
        
        # 1. Find edges on the ground
        ground_edges = self.find_ground_edges(bbox)
        
        if not ground_edges:
            raise ValueError("No ground edges found for screw motion")
        
        # 2. Filter edges by length (must be smaller than gripper size)
        valid_edges = []
        for edge_idx, edge_dir in ground_edges:
                v1_idx, v2_idx = edge_idx
                v1, v2 = vertices[v1_idx], vertices[v2_idx]
                
                # Calculate edge length
                edge_length = LA.norm(v2 - v1)
                
                if edge_length <= self.gripper_max_size:
                    valid_edges.append((edge_idx, edge_dir, edge_length, v1, v2))
        
        if not valid_edges:
            raise ValueError("No valid edges found that fit within gripper size")
        
        # 3. Sort by length (shortest first) and take the two smallest
        valid_edges.sort(key=lambda x: x[2])
        candidate_edges = valid_edges[:min(2, len(valid_edges))]
        
        qs = []
        s_axes = []
        for edge_idx, edge_dir, edge_length, v1, v2 in candidate_edges:
            # Calculate midpoint of the edge
            midpoint = (v1 + v2) / 2
            
            q, s_axis = midpoint, edge_dir/LA.norm(edge_dir)  # q, s_axis
            qs.append(q)
            s_axes.append(s_axis)

        return np.array(qs), np.array(s_axes)
    
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
                                  initial_pose: np.ndarray,
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
                Initial pose of the robot (T, dual quaternion, or pq)
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

        initial_pose = initial_pose.cpu().numpy()
        # if initial_pose is transformation
        if initial_pose.shape == (4, 4):
            initial_pose = pt.dual_quaternion_from_transform(initial_pose)

        # if initial_pose is pq
        elif initial_pose.shape == (7,):
            initial_pose = pt.dual_quaternion_from_pq(initial_pose)

        # if initial_pose is dual quaternion
        elif initial_pose.shape == (8,):
            initial_pose = pt.check_dual_quaternion(initial_pose)

        else:
            raise ValueError(f"Invalid initial pose shape: {initial_pose.shape}, must be (4, 4), (7,), (8,)")
        
        initial_pose = pt.check_dual_quaternion(initial_pose)
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
             bbox: Any, 
             adapter: GeometryAdapter) -> List[Any]:
        """
        Plan trajectory for upright orientation using screw motion.
        
        Args:
            robot: The robot representation
            bbox: The bounding box information
            adapter: Geometry adapter for accessing geometry
            
        Returns:
            List of pose matrices representing the trajectory
        """
        # Get current robot end-effector position
        ee_transform = adapter.compute_forward_kinematics(robot)
        ee_pos = ee_transform[:3, 3]
        
        # Select screw axis and point
        s_axis, q_point = self.select_screw_axis(bbox, ee_pos)
        
        # Set standard parameters for upright orientation
        theta = np.pi/2  # 90 degrees rotation by default
        h = 0.0  # No translation along axis by default
        
        # Generate trajectory
        trajectory = self.generate_screw_trajectory(s_axis, q_point, theta, h, steps=20)
        
        return trajectory