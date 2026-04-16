"""Tests for APIDataset."""

import numpy as np
import pandas as pd
import pytest

from pyaptamer.datasets.dataclasses import APIDataset
from pyaptamer.datasets.dataclasses._base import BaseAptamerDataset


def test_api_dataset_inherits_base():
    assert issubclass(APIDataset, BaseAptamerDataset)


def test_api_dataset_is_not_torch_dataset():
    """Refactor explicitly removes torch.utils.data.Dataset inheritance."""
    import torch.utils.data as torch_data

    assert not issubclass(APIDataset, torch_data.Dataset)


def test_api_dataset_scitype_tag():
    assert APIDataset.get_class_tags()["scitype"] == "APIPairs"


def test_api_dataset_basic_construction():
    ds = APIDataset(x_apta=["ACGU", "UGCA"], x_prot=["MKV", "LKR"], y=[1, 0])
    assert list(ds.load()["aptamer"]) == ["ACGU", "UGCA"]
    assert list(ds.y) == [1, 0]


def test_api_dataset_no_old_kwargs():
    """The old constructor kwargs (apta_max_len, prot_max_len, prot_words,
    split) are removed. Passing them must raise TypeError."""
    with pytest.raises(TypeError):
        APIDataset(x_apta=["A"], x_prot=["M"], y=[1], apta_max_len=100)
    with pytest.raises(TypeError):
        APIDataset(x_apta=["A"], x_prot=["M"], y=[1], split="train")


def test_api_dataset_does_not_encode():
    """Stored data is exactly the input strings, not encoded."""
    ds = APIDataset(x_apta=["ACGU"], x_prot=["MKV"], y=[1])
    df = ds.load()
    assert df["aptamer"].iloc[0] == "ACGU"
    assert df["protein"].iloc[0] == "MKV"
    assert ds.y[0] == 1


def test_from_any_passthrough_for_apidataset():
    ds = APIDataset(x_apta=["A"], x_prot=["M"], y=[1])
    assert APIDataset.from_any(ds) is ds


def test_from_any_with_dataframe():
    df = pd.DataFrame({"aptamer": ["A"], "protein": ["M"]})
    ds = APIDataset.from_any(df, y=[1])
    assert list(ds.load()["aptamer"]) == ["A"]
    assert list(ds.y) == [1]


def test_from_any_with_list_tuples():
    pairs = [("ACGU", "MKV"), ("UGCA", "LKR")]
    ds = APIDataset.from_any(pairs, y=[1, 0])
    assert list(ds.load()["protein"]) == ["MKV", "LKR"]


def test_from_any_with_numpy_pair():
    apta = np.array(["ACGU", "UGCA"])
    prot = np.array(["MKV", "LKR"])
    ds = APIDataset.from_any((apta, prot), y=np.array([1, 0]))
    assert len(ds.load()) == 2


def test_from_any_with_molecule_loader_pair():
    from pathlib import Path

    from pyaptamer.data import MoleculeLoader

    shipped_data_dir = Path("pyaptamer/datasets/data")
    apta_pdb = shipped_data_dir / "1brq.pdb"
    prot_pdb = shipped_data_dir / "5nu7.pdb"
    if not apta_pdb.exists() or not prot_pdb.exists():
        pytest.skip("Need 1brq.pdb and 5nu7.pdb shipped PDB files for this test")
    loader_a = MoleculeLoader(str(apta_pdb))
    loader_b = MoleculeLoader(str(prot_pdb))
    ds = APIDataset.from_any((loader_a, loader_b))
    assert isinstance(ds.load(), pd.DataFrame)
    assert "aptamer" in ds.load().columns


def test_from_any_unsupported_raises():
    with pytest.raises(TypeError):
        APIDataset.from_any("not a valid input")
