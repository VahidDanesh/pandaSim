"""
Configuration module for PandaSim.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any


class GeometryBackend(Enum):
    """Available geometry backend options."""
    GENESIS_AI = auto()
    ROBOTICS_TOOLBOX = auto()


class BoundingBoxType(Enum):
    """Available bounding box computation strategies."""
    AABB = auto()  # Axis-Aligned Bounding Box
    PCA_OBB = auto()  # Principal Component Analysis Oriented Bounding Box


class PlannerType(Enum):
    """Available planner strategy types."""
    PCA = auto()  # Principal Component Analysis based planning
    CONVEX_HULL = auto()  # Convex hull based planning


@dataclass
class SimulationConfig:
    """
    Configuration parameters for PandaSim.
    
    Attributes:
        geometry_backend: Which geometry backend to use
        bbox_type: Which bounding box computation strategy to use
        planner_type: Which planning strategy to use
        backend_params: Additional parameters specific to the geometry backend
        bbox_params: Additional parameters for bounding box computation
        planner_params: Additional parameters for the planning strategy
    """
    geometry_backend: GeometryBackend = GeometryBackend.GENESIS_AI
    bbox_type: BoundingBoxType = BoundingBoxType.PCA_OBB
    planner_type: PlannerType = PlannerType.PCA
    backend_params: Optional[Dict[str, Any]] = None
    bbox_params: Optional[Dict[str, Any]] = None
    planner_params: Optional[Dict[str, Any]] = None