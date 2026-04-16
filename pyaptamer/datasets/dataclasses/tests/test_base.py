"""Tests for BaseAptamerDataset."""

import numpy as np
import pandas as pd
import pytest

from pyaptamer.datasets.dataclasses._base import BaseAptamerDataset
from pyaptamer.datasets.dataclasses.tests.conftest import (
    APTA_A,
    APTA_B,
    PROT_A,
    PROT_B,
)


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
        x_apta=np.array([APTA_A, APTA_B]),
        x_prot=np.array([PROT_A, PROT_B]),
        y=np.array([1, 0]),
    )
    df = ds.load()
    assert list(df.columns) == ["aptamer", "protein"]
    assert list(df["aptamer"]) == [APTA_A, APTA_B]
    assert list(ds.y) == [1, 0]


def test_init_with_lists():
    ds = BaseAptamerDataset(x_apta=[APTA_A, APTA_B], x_prot=[PROT_A, PROT_B], y=[1, 0])
    assert list(ds.load()["protein"]) == [PROT_A, PROT_B]


def test_init_with_pandas_series():
    ds = BaseAptamerDataset(
        x_apta=pd.Series([APTA_A, APTA_B]),
        x_prot=pd.Series([PROT_A, PROT_B]),
    )
    assert ds.y is None
    assert len(ds.load()) == 2


def test_init_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        BaseAptamerDataset(x_apta=[APTA_A], x_prot=[PROT_A, PROT_B])


def test_init_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported type"):
        BaseAptamerDataset(x_apta=42, x_prot=[PROT_A])


def test_load_returns_dataframe():
    ds = BaseAptamerDataset(x_apta=[APTA_A], x_prot=[PROT_A])
    assert isinstance(ds.load(), pd.DataFrame)


def test_y_is_none_when_unlabeled():
    ds = BaseAptamerDataset(x_apta=[APTA_A], x_prot=[PROT_A])
    assert ds.y is None


def test_y_stored_as_numpy_array():
    ds = BaseAptamerDataset(x_apta=[APTA_A], x_prot=[PROT_A], y=[1])
    assert isinstance(ds.y, np.ndarray)
