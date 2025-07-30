"""Loaders for different data structures."""

from pyaptamer.datasets.loader.pfoa_loader import load_pfoa_structure
from pyaptamer.datasets.loader.three_eiy_loader import load_3eiy_structure

__all__ = ["load_pfoa_structure", "load_3eiy_structure"]
