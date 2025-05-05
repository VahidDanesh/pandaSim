"""
Axis-Aligned Bounding Box (AABB) implementation.

This module provides a strategy for computing axis-aligned bounding boxes.
"""
from typing import Any, List, Tuple, Optional
import numpy as np

from pandaSim.geometry.protocols import GeometryAdapter


class AxisAlignedBoundingBox:
    """
    Axis-Aligned Bounding Box representation.
    
    This class represents an axis-aligned bounding box with its center,
    dimensions, and orientation (which is the identity for AABB).
    """
    
    def __init__(self, min_bounds: np.ndarray, max_bounds: np.ndarray):
        """
        Initialize AABB from min and max bounds.
        
        Args:
            min_bounds: Minimum bounds of the box (3D)
            max_bounds: Maximum bounds of the box (3D)
        """
        self._min_bounds = min_bounds
        self._max_bounds = max_bounds
        self._center = (min_bounds + max_bounds) / 2
        self._dimensions = max_bounds - min_bounds
    
    @property
    def center(self) -> np.ndarray:
        """Get the center of the bounding box."""
        return self._center
    
    @property
    def dimensions(self) -> np.ndarray:
        """Get the dimensions (width, height, depth) of the bounding box."""
        return self._dimensions
    
    @property
    def orientation(self) -> np.ndarray:
        """Get the orientation of the bounding box as rotation matrix."""
        return np.eye(3)  # Identity for AABB
    
    @property
    def min_bounds(self) -> np.ndarray:
        """Get the minimum bounds of the box."""
        return self._min_bounds
    
    @property
    def max_bounds(self) -> np.ndarray:
        """Get the maximum bounds of the box."""
        return self._max_bounds


class AABBStrategy:
    """
    Strategy for computing Axis-Aligned Bounding Boxes.
    
    Implements the BoundingBoxStrategy protocol.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize AABB strategy.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
    
    def compute(self, obj: Any, adapter: GeometryAdapter) -> Tuple[AxisAlignedBoundingBox, List[np.ndarray]]:
        """
        Compute axis-aligned bounding box and principal axes for the object.
        
        Args:
            obj: The object representation
            adapter: The geometry adapter to use for accessing the geometry
            
        Returns:
            Tuple containing:
            - The AABB representation
            - List of principal axes (standard basis for AABB)
        """
        # Get vertices from the object using the adapter
        vertices = adapter.get_vertices(obj)
        
        # Compute min and max bounds
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Create AABB
        aabb = AxisAlignedBoundingBox(min_bounds, max_bounds)
        
        # For AABB, principal axes are the standard basis
        axes = [
            np.array([1.0, 0.0, 0.0]),  # X-axis
            np.array([0.0, 1.0, 0.0]),  # Y-axis
            np.array([0.0, 0.0, 1.0])   # Z-axis
        ]
        
        return aabb, axes