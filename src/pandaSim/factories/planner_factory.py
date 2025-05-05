"""
Factory for creating planner strategies.
"""
from typing import Any

from pandaSim.core.config import SimulationConfig, PlannerType
from pandaSim.planning.pca_planner import PCAPlanner
from pandaSim.planning.convex_hull_planner import ConvexHullPlanner


class PlannerFactory:
    """
    Factory for creating planner strategies.
    
    This class implements the Factory pattern to create concrete implementations
    based on configuration.
    """
    
    def create_planner(self, config: SimulationConfig) -> Any:
        """
        Create a planner strategy based on configuration.
        
        Args:
            config: Configuration parameters
            
        Returns:
            Concrete planner strategy implementation
            
        Raises:
            ValueError: If the specified planner type is not supported
        """
        planner_type = config.planner_type
        planner_params = config.planner_params or {}
        
        if planner_type == PlannerType.PCA:
            return PCAPlanner(planner_params)
        elif planner_type == PlannerType.CONVEX_HULL:
            return ConvexHullPlanner(planner_params)
        else:
            raise ValueError(f"Unsupported planner type: {planner_type}")