"""
Protocols for bounding box strategies.

These protocols define the interface that any bounding box strategy must implement.
"""
from typing import Protocol, Any, List, Tuple, runtime_checkable
import numpy as np

from pandaSim.geometry.protocols import GeometryAdapter


@runtime_checkable
class BoundingBox(Protocol):
    """Protocol for bounding box representations."""
    
    @property
    def center(self) -> np.ndarray:
        """Get the center of the bounding box."""
        ...
    
    @property
    def dimensions(self) -> np.ndarray:
        """Get the dimensions (width, height, depth) of the bounding box."""
        ...
    
    @property
    def orientation(self) -> np.ndarray:
        """Get the orientation of the bounding box as rotation matrix."""
        ...


@runtime_checkable
class BoundingBoxStrategy(Protocol):
    """
    Protocol for bounding box computation strategies.
    
    Any bounding box strategy must implement these methods to be compatible with the system.
    """
    
    def compute(self, obj: Any, adapter: GeometryAdapter) -> Tuple[Any, List[np.ndarray]]:
        """
        Compute bounding box and principal axes for the object.
        
        Args:
            obj: The object representation
            adapter: The geometry adapter to use for accessing the geometry
            
        Returns:
            Tuple containing:
            - The bounding box representation
            - List of principal axes (as 3D unit vectors)
        """
        ...