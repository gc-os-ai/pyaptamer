"""Tests for BaseAptamerDataset."""

import numpy as np
import pandas as pd
import pytest

from pyaptamer.datasets.dataclasses._base import BaseAptamerDataset


def test_base_class_default_tags():
    tags = BaseAptamerDataset.get_class_tags()
    assert tags["object_type"] == "dataset"
    assert tags["scitype"] == "APIPairs"
    assert tags["X_inner_mtype"] == ["pd.DataFrame", "list_tuples", "numpy_arrays"]
    assert tags["has_y"] is True


def test_base_class_inherits_from_baseobject():
    from skbase.base import BaseObject

    assert issubclass(BaseAptamerDataset, BaseObject)


def test_init_with_numpy_arrays():
    ds = BaseAptamerDataset(
        x_apta=np.array(["ACGU", "UGCA"]),
        x_prot=np.array(["MKV", "LKR"]),
        y=np.array([1, 0]),
    )
    df = ds.load()
    assert list(df.columns) == ["aptamer", "protein"]
    assert list(df["aptamer"]) == ["ACGU", "UGCA"]
    assert list(ds.y) == [1, 0]


def test_init_with_lists():
    ds = BaseAptamerDataset(x_apta=["ACGU", "UGCA"], x_prot=["MKV", "LKR"], y=[1, 0])
    assert list(ds.load()["protein"]) == ["MKV", "LKR"]


def test_init_with_pandas_series():
    ds = BaseAptamerDataset(
        x_apta=pd.Series(["ACGU", "UGCA"]),
        x_prot=pd.Series(["MKV", "LKR"]),
    )
    assert ds.y is None
    assert len(ds.load()) == 2


def test_init_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        BaseAptamerDataset(x_apta=["ACGU"], x_prot=["MKV", "LKR"])


def test_init_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported type"):
        BaseAptamerDataset(x_apta=42, x_prot=["MKV"])


def test_load_returns_dataframe():
    ds = BaseAptamerDataset(x_apta=["ACGU"], x_prot=["MKV"])
    assert isinstance(ds.load(), pd.DataFrame)


def test_y_is_none_when_unlabeled():
    ds = BaseAptamerDataset(x_apta=["A"], x_prot=["M"])
    assert ds.y is None


def test_y_stored_as_numpy_array():
    ds = BaseAptamerDataset(x_apta=["A"], x_prot=["M"], y=[1])
    assert isinstance(ds.y, np.ndarray)
