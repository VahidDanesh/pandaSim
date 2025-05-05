"""
PandaSim package.
"""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("pandasim")
    except PackageNotFoundError:
        __version__ = "0.0.0"  # Default version if package is not installed
except ImportError:
    __version__ = "0.0.0"  # Fallback for Python < 3.8