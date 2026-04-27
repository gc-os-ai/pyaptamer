"""In-memory data containers for aptamer-related data."""

__author__ = ["nennomp", "siddharth7113"]
__all__ = ["APIDataset", "MaskedDataset"]

from pyaptamer.datasets.dataclasses._api import APIDataset
from pyaptamer.datasets.dataclasses._masked import MaskedDataset
