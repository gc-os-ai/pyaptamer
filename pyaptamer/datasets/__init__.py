"""Contains datasets along with their loaders."""

from pyaptamer.datasets._loaders._one_gnh import load_1gnh_structure
from pyaptamer.datasets._loaders._online_databank import download_and_extract_sequences
from pyaptamer.datasets._loaders._pfoa_loader import load_pfoa_structure

__all__ = [
    "load_pfoa_structure",
    "load_1gnh_structure",
    "download_and_extract_sequences",
]
