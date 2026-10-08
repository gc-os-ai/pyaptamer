"""pyaptamer: Python library for aptamer design."""

from importlib.metadata import version

from pyaptamer._logger import logger

__version__ = version("pyaptamer")

__all__ = ["__version__", "logger"]
