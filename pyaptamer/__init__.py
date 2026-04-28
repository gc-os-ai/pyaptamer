"""pyaptamer: Python library for aptamer design."""

from importlib.metadata import version

try:
    __version__ = version("pyaptamer")
except Exception:
    __version__ = "0.0.1-dev"

from pyaptamer._logger import logger

__version__ = version("pyaptamer")

__all__ = ["__version__", "logger"]
