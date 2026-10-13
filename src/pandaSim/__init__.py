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


# Imported on first access, so that `import pandaSim` does not pull in Genesis or torch
_LAZY = {
    "UprightTask": "pandaSim.core.upright_task",
    "SimulationConfig": "pandaSim.core.config",
    "ResolvedRateController": "pandaSim.control.resolved_rate",
    "QPController": "pandaSim.control.qp",
}

__all__ = ["UprightTask", "SimulationConfig", "ResolvedRateController", "QPController"]


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        value = getattr(import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
