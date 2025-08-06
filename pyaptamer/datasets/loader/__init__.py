"""Loaders for different data structures."""

from pyaptamer.datasets.loader.one_gnh import load_1gnh_structure
from pyaptamer.datasets.loader.pfoa_loader import load_pfoa_structure

__all__ = ["load_pfoa_structure", "load_1gnh_structure"]
