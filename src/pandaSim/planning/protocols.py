"""
Protocols for planning strategies.

These protocols define the interface that any planning strategy must implement.
"""
from typing import Protocol, Any, List, runtime_checkable
import numpy as np

from pandaSim.geometry.protocols import GeometryAdapter


@runtime_checkable
class Trajectory(Protocol):
    """Protocol for trajectory representations."""
    
    @property
    def waypoints(self) -> List[np.ndarray]:
        """Get the waypoints of the trajectory as list of poses."""
        ...
    
    @property
    def durations(self) -> List[float]:
        """Get the durations between waypoints."""
        ...


@runtime_checkable
class PlannerStrategy(Protocol):
    """
    Protocol for planning strategies.
    
    Any planning strategy must implement these methods to be compatible with the system.
    """
    
    def plan(
        self,
        robot: Any,
        bbox: Any,
        adapter: GeometryAdapter
    ) -> List[Any]:
        """
        Plan trajectory to achieve upright orientation.
        
        Args:
            robot: The robot representation
            bbox: The bounding box
            adapter: The geometry adapter to use for accessing the geometry
            
        Returns:
            List of trajectory waypoints
        """
        ...