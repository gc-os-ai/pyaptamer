"""Tests for APIDataset."""

import numpy as np
import pandas as pd
import pytest
from skbase.base import BaseObject

from pyaptamer.datasets.dataclasses import APIDataset
from pyaptamer.datasets.dataclasses.tests.conftest import (
    APTA_A,
    APTA_B,
    PROT_A,
    PROT_B,
)


def test_api_dataset_inherits_baseobject():
    assert issubclass(APIDataset, BaseObject)


def test_api_dataset_is_not_torch_dataset():
    """Refactor explicitly removes torch.utils.data.Dataset inheritance."""
    import torch.utils.data as torch_data

    assert not issubclass(APIDataset, torch_data.Dataset)


def test_api_dataset_scitype_tag():
    assert APIDataset.get_class_tags()["scitype"] == "APIPairs"


def test_api_dataset_basic_construction():
    ds = APIDataset(x_apta=[APTA_A, APTA_B], x_prot=[PROT_A, PROT_B], y=[1, 0])
    assert list(ds.load()["aptamer"]) == [APTA_A, APTA_B]
    assert list(ds.y) == [1, 0]


def test_api_dataset_load_returns_dataframe():
    ds = APIDataset(x_apta=[APTA_A], x_prot=[PROT_A])
    assert isinstance(ds.load(), pd.DataFrame)
    assert list(ds.load().columns) == ["aptamer", "protein"]


def test_api_dataset_no_old_kwargs():
    """The old constructor kwargs (apta_max_len, prot_max_len, prot_words,
    split) are removed. Passing them must raise TypeError."""
    with pytest.raises(TypeError):
        APIDataset(x_apta=[APTA_A], x_prot=[PROT_A], y=[1], apta_max_len=100)
    with pytest.raises(TypeError):
        APIDataset(x_apta=[APTA_A], x_prot=[PROT_A], y=[1], split="train")


def test_api_dataset_does_not_encode():
    """Stored data is exactly the input strings, not encoded."""
    ds = APIDataset(x_apta=[APTA_A], x_prot=[PROT_A], y=[1])
    df = ds.load()
    assert df["aptamer"].iloc[0] == APTA_A
    assert df["protein"].iloc[0] == PROT_A
    assert ds.y[0] == 1


# -- input-coercion tests (moved from test_base.py) -------------------


def test_init_with_numpy_arrays():
    ds = APIDataset(
        x_apta=np.array([APTA_A, APTA_B]),
        x_prot=np.array([PROT_A, PROT_B]),
        y=np.array([1, 0]),
    )
    df = ds.load()
    assert list(df.columns) == ["aptamer", "protein"]
    assert list(df["aptamer"]) == [APTA_A, APTA_B]


def test_init_with_lists():
    ds = APIDataset(x_apta=[APTA_A, APTA_B], x_prot=[PROT_A, PROT_B], y=[1, 0])
    assert list(ds.load()["protein"]) == [PROT_A, PROT_B]


def test_init_with_pandas_series():
    ds = APIDataset(
        x_apta=pd.Series([APTA_A, APTA_B]),
        x_prot=pd.Series([PROT_A, PROT_B]),
    )
    assert ds.y is None
    assert len(ds.load()) == 2


def test_y_coerced_from_single_column_dataframe():
    """When y is a single-column DataFrame (e.g., from load_li2014),
    it should be flattened to a 1D numpy array."""
    y_df = pd.DataFrame({"label": ["positive", "negative"]})
    ds = APIDataset(x_apta=[APTA_A, APTA_B], x_prot=[PROT_A, PROT_B], y=y_df)
    assert ds.y.ndim == 1
    assert list(ds.y) == ["positive", "negative"]


def test_y_coerced_from_2d_array():
    """2D column-vector y should be flattened to 1D."""
    y_2d = np.array([[1], [0]])
    ds = APIDataset(x_apta=[APTA_A, APTA_B], x_prot=[PROT_A, PROT_B], y=y_2d)
    assert ds.y.ndim == 1
    assert list(ds.y) == [1, 0]


def test_init_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        APIDataset(x_apta=[APTA_A], x_prot=[PROT_A, PROT_B])


def test_init_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported type"):
        APIDataset(x_apta=42, x_prot=[PROT_A])


def test_y_is_none_when_unlabeled():
    ds = APIDataset(x_apta=[APTA_A], x_prot=[PROT_A])
    assert ds.y is None
    # Dynamic tag check
    assert ds.get_tags()["has_y"] is False


def test_y_stored_as_numpy_array():
    ds = APIDataset(x_apta=[APTA_A], x_prot=[PROT_A], y=[1])
    assert isinstance(ds.y, np.ndarray)
    # Dynamic tag check
    assert ds.get_tags()["has_y"] is True


# -- from_any tests --------------------------------------------------


def test_from_any_passthrough_for_apidataset():
    ds = APIDataset(x_apta=[APTA_A], x_prot=[PROT_A], y=[1])
    assert APIDataset.from_any(ds) is ds


def test_from_any_with_dataframe():
    df = pd.DataFrame({"aptamer": [APTA_A], "protein": [PROT_A]})
    ds = APIDataset.from_any(df, y=[1])
    assert list(ds.load()["aptamer"]) == [APTA_A]
    assert list(ds.y) == [1]


def test_from_any_with_custom_column_names():
    """DataFrame input with non-default column names is renamed via
    ``apta_col`` / ``prot_col`` before coercion."""
    df = pd.DataFrame({"my_apta": [APTA_A, APTA_B], "my_prot": [PROT_A, PROT_B]})
    ds = APIDataset.from_any(df, y=[1, 0], apta_col="my_apta", prot_col="my_prot")
    # Internal storage uses canonical names regardless of input column names.
    assert list(ds.load().columns) == ["aptamer", "protein"]
    assert list(ds.load()["aptamer"]) == [APTA_A, APTA_B]
    assert list(ds.load()["protein"]) == [PROT_A, PROT_B]


def test_from_any_default_column_names_unchanged():
    """Defaults match canonical names — existing callers untouched."""
    df = pd.DataFrame({"aptamer": [APTA_A], "protein": [PROT_A]})
    ds = APIDataset.from_any(df)
    assert list(ds.load().columns) == ["aptamer", "protein"]


def test_from_any_with_list_tuples():
    pairs = [(APTA_A, PROT_A), (APTA_B, PROT_B)]
    ds = APIDataset.from_any(pairs, y=[1, 0])
    assert list(ds.load()["protein"]) == [PROT_A, PROT_B]


def test_from_any_with_numpy_pair():
    apta = np.array([APTA_A, APTA_B])
    prot = np.array([PROT_A, PROT_B])
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
