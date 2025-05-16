"""
Factory for creating motion controllers.
"""
from typing import Any, Dict, Optional

from pandaSim.control.protocols import MotionController
from pandaSim.control.resolved_rate import ResolvedRateController
from pandaSim.geometry.protocols import GeometryAdapter


class MotionControllerFactory:
    """
    Factory for creating motion controllers.
    
    This class implements the Factory pattern to create concrete implementations
    based on configuration.
    """
    
    def create_controller(
        self,
        adapter: GeometryAdapter,
        controller_type: str = "resolved_rate",
        config: Optional[Dict[str, Any]] = None
    ) -> MotionController:
        """
        Create a motion controller based on configuration.
        
        Args:
            adapter: Geometry adapter for the controller to use
            controller_type: Type of controller ("resolved_rate" for Resolved-Rate Motion Control)
            config: Configuration parameters
            
        Returns:
            Concrete motion controller implementation
            
        Raises:
            ValueError: If the specified controller type is not supported
        """
        controller_params = config or {}
        
        if controller_type.lower() == "resolved_rate":
            return ResolvedRateController(adapter=adapter, **controller_params)
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}") 