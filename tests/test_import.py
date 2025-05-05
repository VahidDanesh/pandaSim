"""
Test module imports and basic functionality.
"""
import pytest


def test_package_imports():
    """Test that the package imports correctly and __version__ is defined."""
    import pandaSim
    assert hasattr(pandaSim, "__version__")
    assert isinstance(pandaSim.__version__, str)


def test_core_imports():
    """Test core module imports."""
    from pandaSim import UprightTask, SimulationConfig
    from pandaSim.core.config import GeometryBackend, BoundingBoxType, PlannerType
    
    # Create a basic configuration
    config = SimulationConfig(
        geometry_backend=GeometryBackend.GENESIS_AI,
        bbox_type=BoundingBoxType.PCA_OBB,
        planner_type=PlannerType.PCA
    )
    
    # Create an instance of UprightTask
    task = UprightTask(config)
    
    # Check that the task instance was created successfully
    assert isinstance(task, UprightTask)
    assert task.config == config


def test_geometry_adapters():
    """Test geometry adapter imports and basic functionality."""
    from pandaSim.geometry.genesis_adapter import GenesisAdapter
    from pandaSim.geometry.rtb_adapter import RoboticsToolboxAdapter
    from pandaSim.geometry.protocols import GeometryAdapter
    
    # Create adapter instances
    genesis = GenesisAdapter()
    rtb = RoboticsToolboxAdapter()
    
    # Check that they implement the protocol
    assert isinstance(genesis, GeometryAdapter)
    assert isinstance(rtb, GeometryAdapter)


def test_fitting_strategies():
    """Test bounding box strategy imports and basic functionality."""
    from pandaSim.fitting.aabb import AABBStrategy, AxisAlignedBoundingBox
    from pandaSim.fitting.pca_obb import PCAOBBStrategy, OrientedBoundingBox
    from pandaSim.fitting.protocols import BoundingBoxStrategy
    
    # Create strategy instances
    aabb_strategy = AABBStrategy()
    pca_strategy = PCAOBBStrategy()
    
    # Check that they implement the protocol
    assert isinstance(aabb_strategy, BoundingBoxStrategy)
    assert isinstance(pca_strategy, BoundingBoxStrategy)


def test_planner_strategies():
    """Test planner strategy imports and basic functionality."""
    from pandaSim.planning.pca_planner import PCAPlanner, PCATrajectory
    from pandaSim.planning.convex_hull_planner import ConvexHullPlanner, ConvexHullTrajectory
    from pandaSim.planning.protocols import PlannerStrategy
    
    # Create planner instances
    pca_planner = PCAPlanner()
    ch_planner = ConvexHullPlanner()
    
    # Check that they implement the protocol
    assert isinstance(pca_planner, PlannerStrategy)
    assert isinstance(ch_planner, PlannerStrategy)


def test_factories():
    """Test factory imports and basic functionality."""
    from pandaSim.core.config import SimulationConfig, GeometryBackend, BoundingBoxType, PlannerType
    from pandaSim.factories.geometry_factory import GeometryFactory
    from pandaSim.factories.planner_factory import PlannerFactory
    from pandaSim.geometry.protocols import GeometryAdapter
    from pandaSim.fitting.protocols import BoundingBoxStrategy
    from pandaSim.planning.protocols import PlannerStrategy
    
    # Create a basic configuration
    config = SimulationConfig(
        geometry_backend=GeometryBackend.GENESIS_AI,
        bbox_type=BoundingBoxType.PCA_OBB,
        planner_type=PlannerType.PCA
    )
    
    # Create factories
    geometry_factory = GeometryFactory()
    planner_factory = PlannerFactory()
    
    # Create components
    adapter = geometry_factory.create_adapter(config)
    bbox_strategy = geometry_factory.create_bbox_strategy(config)
    planner = planner_factory.create_planner(config)
    
    # Check that they implement the respective protocols
    assert isinstance(adapter, GeometryAdapter)
    assert isinstance(bbox_strategy, BoundingBoxStrategy)
    assert isinstance(planner, PlannerStrategy)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])