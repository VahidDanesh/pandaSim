"""pandaSim: planning and executing primitive motions of a box with environmental contact on a Franka Panda."""

try:
    from importlib.metadata import PackageNotFoundError, version
    try:
        __version__ = version("pandasim")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except ImportError:
    __version__ = "0.0.0"
