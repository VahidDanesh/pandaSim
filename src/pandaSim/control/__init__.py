"""
Control module for motion controllers.

This module provides interfaces and implementations for robot motion control.
"""

from pandaSim.control.protocols import MotionController
from pandaSim.control.config import ExecutionConfig
from pandaSim.control.data_collector import DataCollector, ExecutionResult

# Imported on first access, so that this package does not pull in Genesis or panda-py
_LAZY = {
    "ResolvedRateController": "pandaSim.control.resolved_rate",
    "QPController": "pandaSim.control.qp",
    "MotionControllerFactory": "pandaSim.control.factory",
    "RobotExecutor": "pandaSim.control.robot_executor",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        value = getattr(import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MotionController", 
    "ResolvedRateController", 
    "QPController", 
    "MotionControllerFactory",
    "ExecutionConfig",
    "DataCollector",
    "ExecutionResult", 
    "RobotExecutor"
] 