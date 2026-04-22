"""pyaptamer: Python library for aptamer design."""

from importlib.metadata import version

from pyaptamer._config import config

__version__ = version("pyaptamer")
logger = config.logger

__all__ = ["__version__", "config", "logger"]
