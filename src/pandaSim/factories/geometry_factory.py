"""
Factory for creating geometry adapters and bounding box strategies.
"""
from typing import Any

from pandaSim.core.config import SimulationConfig, GeometryBackend, BoundingBoxType
from pandaSim.geometry.genesis_adapter import GenesisAdapter
from pandaSim.geometry.rtb_adapter import RoboticsToolboxAdapter
from pandaSim.fitting.aabb import AABBStrategy
from pandaSim.fitting.pca_obb import PCAOBBStrategy


class GeometryFactory:
    """
    Factory for creating geometry adapters and bounding box strategies.
    
    This class implements the Factory pattern to create concrete implementations
    based on configuration.
    """
    
    def create_adapter(self, config: SimulationConfig) -> Any:
        """
        Create a geometry adapter based on configuration.
        
        Args:
            config: Configuration parameters
            
        Returns:
            Concrete geometry adapter implementation
            
        Raises:
            ValueError: If the specified backend is not supported
        """
        backend = config.geometry_backend
        backend_params = config.backend_params or {}
        
        if backend == GeometryBackend.GENESIS_AI:
            return GenesisAdapter(backend_params)
        elif backend == GeometryBackend.ROBOTICS_TOOLBOX:
            return RoboticsToolboxAdapter(backend_params)
        else:
            raise ValueError(f"Unsupported geometry backend: {backend}")
    
    def create_bbox_strategy(self, config: SimulationConfig) -> Any:
        """
        Create a bounding box strategy based on configuration.
        
        Args:
            config: Configuration parameters
            
        Returns:
            Concrete bounding box strategy implementation
            
        Raises:
            ValueError: If the specified bbox type is not supported
        """
        bbox_type = config.bbox_type
        bbox_params = config.bbox_params or {}
        
        if bbox_type == BoundingBoxType.AABB:
            return AABBStrategy(bbox_params)
        elif bbox_type == BoundingBoxType.PCA_OBB:
            return PCAOBBStrategy(bbox_params)
        else:
            raise ValueError(f"Unsupported bounding box type: {bbox_type}")