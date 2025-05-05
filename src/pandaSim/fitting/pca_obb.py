"""
Principal Component Analysis Oriented Bounding Box (PCA-OBB) implementation.

This module provides a strategy for computing oriented bounding boxes based on PCA.
"""
from typing import Any, List, Tuple, Optional
import numpy as np

from pandaSim.geometry.protocols import GeometryAdapter


class OrientedBoundingBox:
    """
    Oriented Bounding Box representation.
    
    This class represents an oriented bounding box with its center,
    dimensions, and orientation.
    """
    
    def __init__(self, center: np.ndarray, dimensions: np.ndarray, orientation: np.ndarray):
        """
        Initialize OBB.
        
        Args:
            center: Center of the box (3D)
            dimensions: Dimensions of the box (width, height, depth)
            orientation: Orientation as 3x3 rotation matrix
        """
        self._center = center
        self._dimensions = dimensions
        self._orientation = orientation
    
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
        return self._orientation


class PCAOBBStrategy:
    """
    Strategy for computing Oriented Bounding Boxes using PCA.
    
    Implements the BoundingBoxStrategy protocol.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize PCA-OBB strategy.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
    
    def compute(self, obj: Any, adapter: GeometryAdapter) -> Tuple[OrientedBoundingBox, List[np.ndarray]]:
        """
        Compute oriented bounding box and principal axes for the object using PCA.
        
        Args:
            obj: The object representation
            adapter: The geometry adapter to use for accessing the geometry
            
        Returns:
            Tuple containing:
            - The OBB representation
            - List of principal axes (eigenvectors from PCA)
        """
        # Get vertices from the object using the adapter
        vertices = adapter.get_vertices(obj)
        
        # Compute center of mass (mean of vertices)
        center = np.mean(vertices, axis=0)
        
        # Center the vertices
        centered_vertices = vertices - center
        
        # Perform PCA (compute covariance matrix and its eigenvectors)
        cov_matrix = np.cov(centered_vertices, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # These are the principal axes
        axes = [eigenvectors[:, i] for i in range(3)]
        
        # Project vertices onto principal axes
        projected = np.dot(centered_vertices, eigenvectors)
        
        # Compute min and max along each principal axis
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        
        # Compute dimensions
        dimensions = max_proj - min_proj
        
        # Create OBB
        obb = OrientedBoundingBox(center, dimensions, eigenvectors)
        
        return obb, axes