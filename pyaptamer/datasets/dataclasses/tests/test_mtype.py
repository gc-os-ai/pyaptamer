"""Tests for the mtype dispatcher."""

import numpy as np
import pandas as pd
import pytest

from pyaptamer.datasets.dataclasses._mtype import (
    INPUT_ONLY_MTYPES,
    SUPPORTED_MTYPES,
    convert_to,
)
from pyaptamer.datasets.dataclasses.tests.conftest import (
    APTA_A,
    APTA_B,
    PROT_A,
    PROT_B,
)


def test_supported_mtypes_contains_three_canonical_names():
    assert "pd.DataFrame" in SUPPORTED_MTYPES
    assert "list_tuples" in SUPPORTED_MTYPES
    assert "numpy_arrays" in SUPPORTED_MTYPES
    assert len(SUPPORTED_MTYPES) == 3


def test_input_only_mtypes_contains_molecule_loader_pair():
    assert "MoleculeLoader_pair" in INPUT_ONLY_MTYPES


@pytest.fixture
def df_two_pairs():
    return pd.DataFrame({"aptamer": [APTA_A, APTA_B], "protein": [PROT_A, PROT_B]})


def test_convert_df_to_list_tuples(df_two_pairs):
    result = convert_to(df_two_pairs, to_mtype="list_tuples")
    assert result == [(APTA_A, PROT_A), (APTA_B, PROT_B)]


def test_convert_df_to_numpy_arrays(df_two_pairs):
    apta, prot = convert_to(df_two_pairs, to_mtype="numpy_arrays")
    assert isinstance(apta, np.ndarray) and isinstance(prot, np.ndarray)
    assert list(apta) == [APTA_A, APTA_B]
    assert list(prot) == [PROT_A, PROT_B]


def test_convert_list_tuples_to_df():
    pairs = [(APTA_A, PROT_A), (APTA_B, PROT_B)]
    result = convert_to(pairs, to_mtype="pd.DataFrame")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["aptamer", "protein"]
    assert list(result["aptamer"]) == [APTA_A, APTA_B]


def test_convert_numpy_to_df():
    apta = np.array([APTA_A, APTA_B])
    prot = np.array([PROT_A, PROT_B])
    result = convert_to((apta, prot), to_mtype="pd.DataFrame")
    assert isinstance(result, pd.DataFrame)
    assert list(result["protein"]) == [PROT_A, PROT_B]


def test_convert_to_same_mtype_is_passthrough(df_two_pairs):
    result = convert_to(df_two_pairs, to_mtype="pd.DataFrame")
    assert result is df_two_pairs


def test_convert_to_unknown_mtype_raises(df_two_pairs):
    with pytest.raises(ValueError, match="Unknown to_mtype"):
        convert_to(df_two_pairs, to_mtype="not_a_real_mtype")


def test_convert_unsupported_input_raises():
    with pytest.raises(TypeError, match="Cannot convert"):
        convert_to(42, to_mtype="pd.DataFrame")


def test_convert_molecule_loader_pair_to_df():
    """MoleculeLoader_pair is input-only; coerces to pd.DataFrame."""
    from pathlib import Path

    from pyaptamer.data import MoleculeLoader
    from pyaptamer.datasets.dataclasses._mtype import coerce_input

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


def test_coerce_input_passthrough_for_supported_mtypes(df_two_pairs):
    from pyaptamer.datasets.dataclasses._mtype import coerce_input

    assert coerce_input(df_two_pairs) is df_two_pairs


def test_coerce_input_raises_on_unknown_type():
    from pyaptamer.datasets.dataclasses._mtype import coerce_input

    with pytest.raises(TypeError):
        coerce_input(42)
