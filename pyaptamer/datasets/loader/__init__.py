"""Loaders for different data structures."""

from pyaptamer.datasets.loader.one_ghn import load_1ghn_structure
from pyaptamer.datasets.loader.pfoa_loader import load_pfoa_structure

__all__ = ["load_pfoa_structure", "load_1ghn_structure"]
