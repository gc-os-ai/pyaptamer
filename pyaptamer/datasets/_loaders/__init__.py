"""Loaders for different data structures."""

from pyaptamer.datasets._loaders._one_gnh import load_1gnh_structure
from pyaptamer.datasets._loaders._pfoa_loader import load_pfoa_structure
from pyaptamer.datasets._loaders.load_aptamer_interactions import (
    load_aptadb,
    load_interactions,
)

__all__ = [
    "load_pfoa_structure",
    "load_1gnh_structure",
    "load_aptadb",
    "load_interactions",
]
