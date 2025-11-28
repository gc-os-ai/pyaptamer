"""Contains datasets along with their loaders."""

from pyaptamer.datasets._loaders import (
    load_1brq,
    load_1brq_structure,
    load_1gnh,
    load_1gnh_structure,
    load_5nu7,
    load_5nu7_structure,
    load_aptacom_full,
    load_aptacom_xy,
    load_csv_dataset,
    load_from_rcsb,
    load_hf_dataset,
)

__all__ = [
    "load_aptacom_full",
    "load_aptacom_xy",
    "load_csv_dataset",
    "load_hf_dataset",
    "load_from_rcsb",
    "mol_loader",
    "structure_loader",
    "load_1brq",
    "load_1brq_structure",
    "load_5nu7",
    "load_5nu7_structure",
    "load_1gnh",
    "load_1gnh_structure",
]
