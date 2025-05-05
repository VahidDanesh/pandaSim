"""
Protocols for geometry adapters.

These protocols define the interface that any geometry backend adapter must implement.
"""
from typing import Protocol, Any, List, Tuple, Optional, runtime_checkable
import numpy as np


@runtime_checkable
class GeometryAdapter(Protocol):
    """
    Protocol for geometry backend adapters.
    
    Any adapter must implement these methods to be compatible with the system.
    """
    
    def load(self, file_path: str) -> Any:
        """
        Load geometry data from file.
        
        Args:
            file_path: Path to the geometry file (mesh, point cloud, etc.)
            
        Returns:
            The loaded geometry representation
        """
        ...
    
    def get_vertices(self, obj: Any) -> np.ndarray:
        """
        Extract vertices from the geometry.
        
        Args:
            obj: The geometry representation
            
        Returns:
            Numpy array of vertices (Nx3)
        """
        ...
    
    def get_faces(self, obj: Any) -> Optional[np.ndarray]:
        """
        Extract faces from the geometry, if available.
        
        Args:
            obj: The geometry representation
            
        Returns:
            Numpy array of face indices or None if not applicable
        """
        ...
    
    def transform(self, obj: Any, transformation: np.ndarray) -> Any:
        """
        Apply transformation to the geometry.
        
        Args:
            obj: The geometry representation
            transformation: 4x4 transformation matrix
            
        Returns:
            Transformed geometry
        """
        ...
    
    def compute_center_of_mass(self, obj: Any) -> np.ndarray:
        """
        Compute center of mass of the geometry.
        
        Args:
            obj: The geometry representation
            
        Returns:
            3D coordinates of the center of mass
        """
        ...
    
    def compute_volume(self, obj: Any) -> float:
        """
        Compute volume of the geometry.
        
        Args:
            obj: The geometry representation
            
        Returns:
            Volume of the geometry
        """
        ...