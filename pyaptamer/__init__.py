"""pyaptamer: Python library for aptamer design."""

from importlib.metadata import version

from pyaptamer.trafos.encode import (
    KMerEncoder,
    PairsToFeaturesTransformer,
    PSeAACEncoder,
)

__version__ = version("pyaptamer")

__all__ = [
    "KMerEncoder",
    "PSeAACEncoder",
    "PairsToFeaturesTransformer",
]
from pyaptamer._logger import logger

__version__ = version("pyaptamer")

__all__ = ["__version__", "logger"]
