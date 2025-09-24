"""Contains datasets along with their loaders."""

from pyaptamer.datasets._loaders._one_gnh import load_1gnh_structure
from pyaptamer.datasets._loaders._online_databank import load_from_rcsb
from pyaptamer.datasets._loaders._pfoa_loader import load_pfoa_structure
from pyaptamer.datasets._loaders._aptacom_loader import load_aptacom

__all__ = [
    "load_pfoa_structure",
    "load_1gnh_structure",
    "load_from_rcsb",
    "load_aptacom",
]
