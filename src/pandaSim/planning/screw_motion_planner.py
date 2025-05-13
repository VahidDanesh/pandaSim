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
        
    def generate_screw_trajectory(self, 
                                  s_axis: np.ndarray, 
                                  q: np.ndarray, 
                                  theta: float = np.pi/2, 
                                  h: float = 0.0, 
                                  steps: int = 20) -> List[np.ndarray]:
        """
        Generate trajectory waypoints using screw motion.
        
        Args:
            s_axis: Unit vector along screw axis
            q_point: Point on screw axis
            theta: Total rotation angle (default: 90 degrees)
            h: Screw pitch (default: 0 = pure rotation)
            steps: Number of waypoints to generate
            
        Returns:
            List of pose matrices representing the trajectory
        """
        ...
    
    def plan(self, robot: Any, bbox: Any, adapter: GeometryAdapter) -> List[Any]:
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