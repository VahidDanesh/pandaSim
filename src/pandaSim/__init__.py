"""
PandaSim - A SOLID-compliant, backend-agnostic framework for robotic simulation.

This package provides interfaces and implementations for geometry adapters,
bounding-box strategies, and planning strategies for robotic tasks.
"""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("pandasim")
    except PackageNotFoundError:
        __version__ = "0.0.0"  # Default version if package is not installed
except ImportError:
    __version__ = "0.0.0"  # Fallback for Python < 3.8


from pandaSim.core.upright_task import UprightTask
from pandaSim.core.config import SimulationConfig

__all__ = ["UprightTask", "SimulationConfig"]