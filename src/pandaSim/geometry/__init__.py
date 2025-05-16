"""Geometry module for PandaSim.

Provides adapters and utility functions for handling geometry in different backends.
"""

from pandaSim.geometry.protocols import GeometryAdapter
from pandaSim.geometry.utils import convert_pose

__all__ = ["GeometryAdapter", "convert_pose"] 