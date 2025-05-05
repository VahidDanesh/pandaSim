"""
Core implementation of the UprightTask, which orchestrates the full pipeline.
"""
from typing import Optional, Any, List, Tuple

from pandaSim.core.config import SimulationConfig
from pandaSim.geometry.protocols import GeometryAdapter
from pandaSim.fitting.protocols import BoundingBoxStrategy
from pandaSim.planning.protocols import PlannerStrategy
from pandaSim.factories.geometry_factory import GeometryFactory
from pandaSim.factories.planner_factory import PlannerFactory


class UprightTask:
    """
    Main entry point for upright task simulation.
    
    This class orchestrates the full pipeline:
    1. Geometry adaptation (backend-agnostic handling of mesh/point cloud)
    2. Bounding box fitting (determining principal axes for the object)
    3. Planning (generating trajectory for upright orientation)
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        geometry_adapter: Optional[GeometryAdapter] = None,
        bbox_strategy: Optional[BoundingBoxStrategy] = None,
        planner_strategy: Optional[PlannerStrategy] = None,
    ):
        """
        Initialize UprightTask with appropriate components.
        
        Args:
            config: Configuration parameters for the simulation
            geometry_adapter: Optional adapter for the specific geometry backend
            bbox_strategy: Optional strategy for bounding box computation
            planner_strategy: Optional strategy for planning
        """
        self.config = config
        
        # Use factories to create components if not provided
        geometry_factory = GeometryFactory()
        planner_factory = PlannerFactory()
        
        self.geometry_adapter = geometry_adapter or geometry_factory.create_adapter(config)
        self.bbox_strategy = bbox_strategy or geometry_factory.create_bbox_strategy(config)
        self.planner_strategy = planner_strategy or planner_factory.create_planner(config)
    
    def load_object(self, object_path: str) -> Any:
        """
        Load object from file using the geometry adapter.
        
        Args:
            object_path: Path to the object file (mesh, point cloud, etc.)
            
        Returns:
            The loaded object representation
        """
        return self.geometry_adapter.load(object_path)
    
    def compute_bounding_box(self, obj: Any) -> Tuple[Any, List[Any]]:
        """
        Compute the bounding box and principal axes for the object.
        
        Args:
            obj: The object representation
            
        Returns:
            Tuple containing the bounding box and principal axes
        """
        return self.bbox_strategy.compute(obj, self.geometry_adapter)
    
    def plan_upright_orientation(self, obj: Any, bbox: Any, axes: List[Any]) -> List[Any]:
        """
        Plan trajectory to achieve upright orientation.
        
        Args:
            obj: The object representation
            bbox: The bounding box
            axes: The principal axes
            
        Returns:
            List of trajectory waypoints
        """
        return self.planner_strategy.plan(obj, bbox, axes, self.geometry_adapter)
    
    def execute(self, object_path: str) -> List[Any]:
        """
        Execute the full upright task pipeline.
        
        Args:
            object_path: Path to the object file
            
        Returns:
            List of trajectory waypoints for achieving upright orientation
        """
        obj = self.load_object(object_path)
        bbox, axes = self.compute_bounding_box(obj)
        trajectory = self.plan_upright_orientation(obj, bbox, axes)
        return trajectory