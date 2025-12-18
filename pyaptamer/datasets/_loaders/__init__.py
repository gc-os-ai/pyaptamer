"""Loaders for different data structures."""

from pyaptamer.datasets._loaders._1brq import (
    load_1brq,
    load_1brq_structure,
)
from pyaptamer.datasets._loaders._1gnh import (
    load_1gnh,
    load_1gnh_structure,
)
from pyaptamer.datasets._loaders._5nu7 import (
    load_5nu7,
    load_5nu7_structure,
)
from pyaptamer.datasets._loaders._aptacom_loader import (
    load_aptacom_full,
    load_aptacom_x_y,
)
from pyaptamer.datasets._loaders._csv_loader import load_csv_dataset
from pyaptamer.datasets._loaders._hf_loader import load_hf_dataset
from pyaptamer.datasets._loaders._online_databank import load_from_rcsb
from pyaptamer.datasets._loaders._pfoa import (
    load_pfoa,
    load_pfoa_structure,
)

__all__ = [
    "load_aptacom_full",
    "load_aptacom_x_y",
    "load_csv_dataset",
    "load_from_rcsb",
    "load_hf_dataset",
    "load_1gnh",
    "load_1gnh_structure",
    "load_1brq",
    "load_1brq_structure",
    "load_5nu7",
    "load_5nu7_structure",
    "load_pfoa",
    "load_pfoa_structure",
]
