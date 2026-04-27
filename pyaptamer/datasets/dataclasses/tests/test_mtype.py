"""Tests for input coercion."""

import numpy as np
import pandas as pd
import pytest

from pyaptamer.datasets.dataclasses._mtype import coerce_input
from pyaptamer.datasets.dataclasses.tests.conftest import (
    APTA_A,
    APTA_B,
    PROT_A,
    PROT_B,
)


@pytest.fixture
def df_two_pairs():
    return pd.DataFrame({"aptamer": [APTA_A, APTA_B], "protein": [PROT_A, PROT_B]})


def test_coerce_dataframe_passthrough(df_two_pairs):
    """A DataFrame is returned unchanged (identity)."""
    assert coerce_input(df_two_pairs) is df_two_pairs


def test_coerce_list_tuples():
    pairs = [(APTA_A, PROT_A), (APTA_B, PROT_B)]
    result = coerce_input(pairs)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["aptamer", "protein"]
    assert list(result["aptamer"]) == [APTA_A, APTA_B]


def test_coerce_numpy_pair():
    apta = np.array([APTA_A, APTA_B])
    prot = np.array([PROT_A, PROT_B])
    result = coerce_input((apta, prot))
    assert isinstance(result, pd.DataFrame)
    assert list(result["protein"]) == [PROT_A, PROT_B]


def test_coerce_molecule_loader_pair():
    """MoleculeLoader pair coerces to DataFrame via .to_df_seq()."""
    from pathlib import Path

    from pyaptamer.data import MoleculeLoader

    shipped_data_dir = Path("pyaptamer/datasets/data")
    apta_pdb = shipped_data_dir / "1brq.pdb"
    prot_pdb = shipped_data_dir / "5nu7.pdb"
    if not apta_pdb.exists() or not prot_pdb.exists():
        pytest.skip("Need 1brq.pdb and 5nu7.pdb shipped PDB files for this test")

    loader_a = MoleculeLoader(str(apta_pdb))
    loader_b = MoleculeLoader(str(prot_pdb))
    df = coerce_input((loader_a, loader_b))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["aptamer", "protein"]
    assert len(df) > 0


def test_coerce_unknown_type_raises():
    with pytest.raises(TypeError, match="Cannot coerce"):
        coerce_input(42)


def test_coerce_unknown_tuple_raises():
    """A 2-tuple of unsupported types must raise."""
    with pytest.raises(TypeError, match="Cannot coerce"):
        coerce_input(("not an array", "also not"))
