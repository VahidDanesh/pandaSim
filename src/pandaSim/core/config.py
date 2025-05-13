"""
Configuration classes for simulation setup.
"""
from enum import Enum, auto
from typing import Optional, Dict, Any


class GeometryBackend(Enum):
    """Available geometry backend options."""
    GENESIS_AI = auto()
    ROBOTICS_TOOLBOX = auto() # TODO: Implement robotics toolbox backend


class BoundingBoxType(Enum):
    """Available bounding box computation strategies."""
    AABB = auto()  # Axis-Aligned Bounding Box
    PCA_OBB = auto()  # TODO: Implement Principal Component Analysis Oriented Bounding Box


class PlannerType(Enum):
    """Available planner strategy types."""
    SCREW_MOTION = auto()  # Screw motion based planning


class SimulationConfig:
    """Configuration parameters for the simulation."""
    
    def __init__(
        self,
        geometry_backend: GeometryBackend = GeometryBackend.GENESIS_AI,
        bbox_type: BoundingBoxType = BoundingBoxType.AABB,
        planner_type: PlannerType = PlannerType.SCREW_MOTION,
        geometry_config: Optional[Dict[str, Any]] = None,
        bbox_config: Optional[Dict[str, Any]] = None,
        planner_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize simulation configuration.
        
        Args:
            geometry_backend: Which geometry backend to use
            bbox_type: Which bounding box computation strategy to use
            planner_type: Which planning strategy to use
            geometry_config: Configuration for geometry adapter
            bbox_config: Configuration for bounding box strategy
            planner_config: Configuration for planner strategy
        """
        self.geometry_backend = geometry_backend
        self.bbox_type = bbox_type
        self.planner_type = planner_type
        self.geometry_config = geometry_config
        self.bbox_config = bbox_config
        self.planner_config = planner_config