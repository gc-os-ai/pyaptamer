"""Contains datasets along with their loaders."""

from pyaptamer.datasets._loaders._aptacom_loader import (
    load_aptacom_full,
    load_aptacom_xy,
)
from pyaptamer.datasets._loaders._csv_loader import load_csv_dataset
from pyaptamer.datasets._loaders._hf_loader import load_hf_dataset
from pyaptamer.datasets._loaders._load_aptamer import (
    load_aptadb,
    load_encoders,
)
from pyaptamer.datasets._loaders._one_gnh import load_1gnh, load_1gnh_structure
from pyaptamer.datasets._loaders._online_databank import load_from_rcsb
from pyaptamer.datasets._loaders._pfoa import load_pfoa, load_pfoa_structure

__all__ = [
    "load_aptacom_full",
    "load_aptacom_x_y",
    "load_csv_dataset",
    "load_hf_dataset",
    "load_pfoa",
    "load_pfoa_structure",
    "load_1gnh",
    "load_1gnh_structure",
    "load_from_rcsb",
    "load_csv_dataset",
    "load_aptadb",
    "load_encoders",
]
