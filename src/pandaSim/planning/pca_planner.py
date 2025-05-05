"""
PCA-based planner implementation.

This module provides a planning strategy based on principal component analysis.
"""
from typing import Any, List, Optional
import numpy as np

from pandaSim.geometry.protocols import GeometryAdapter


class PCATrajectory:
    """
    Trajectory representation for PCA-based planning.
    """
    
    def __init__(self, waypoints: List[np.ndarray], durations: List[float]):
        """
        Initialize trajectory.
        
        Args:
            waypoints: List of poses (4x4 transformation matrices)
            durations: List of durations between waypoints
        """
        self._waypoints = waypoints
        self._durations = durations
    
    @property
    def waypoints(self) -> List[np.ndarray]:
        """Get the waypoints of the trajectory."""
        return self._waypoints
    
    @property
    def durations(self) -> List[float]:
        """Get the durations between waypoints."""
        return self._durations


class PCAPlanner:
    """
    Planning strategy based on Principal Component Analysis.
    
    Implements the PlannerStrategy protocol.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize PCA planner.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.num_waypoints = self.config.get("num_waypoints", 5)
        self.duration_per_waypoint = self.config.get("duration_per_waypoint", 1.0)
    
    def plan(
        self,
        obj: Any,
        bbox: Any,
        axes: List[np.ndarray],
        adapter: GeometryAdapter
    ) -> List[PCATrajectory]:
        """
        Plan trajectory to achieve upright orientation based on PCA.
        
        Args:
            obj: The object representation
            bbox: The bounding box
            axes: The principal axes
            adapter: The geometry adapter to use for accessing the geometry
            
        Returns:
            List of trajectory waypoints
        """
        # Determine the upright axis (usually the longest axis for stability)
        dimensions = bbox.dimensions
        longest_axis_idx = np.argmax(dimensions)
        
        # Get the corresponding principal axis
        upright_axis = axes[longest_axis_idx]
        
        # Determine target orientation (aligning upright axis with gravity)
        z_axis = np.array([0.0, 0.0, 1.0])  # Gravity direction (up)
        
        # Compute rotation to align upright axis with z-axis
        rotation_axis = np.cross(upright_axis, z_axis)
        
        # If axes are already aligned, use a different axis for rotation
        if np.allclose(rotation_axis, 0.0):
            if np.allclose(upright_axis, z_axis):
                # Already aligned, no rotation needed
                rotation_angle = 0.0
            else:
                # Completely anti-aligned, rotate around x-axis
                rotation_axis = np.array([1.0, 0.0, 0.0])
                rotation_angle = np.pi
        else:
            # Normalize rotation axis
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # Compute rotation angle
            dot_product = np.dot(upright_axis, z_axis)
            rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Generate waypoints by interpolating the rotation
        waypoints = []
        durations = []
        
        for i in range(self.num_waypoints):
            # Interpolate rotation angle
            t = i / (self.num_waypoints - 1)
            angle = t * rotation_angle
            
            # Compute rotation matrix using Rodrigues' formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = bbox.center
            
            waypoints.append(T)
            
            if i > 0:
                durations.append(self.duration_per_waypoint)
        
        return [PCATrajectory(waypoints, durations)]