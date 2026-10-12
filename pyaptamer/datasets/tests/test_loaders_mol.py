"""Tests for the molecule dataset loaders (load_1gnh, load_1brq, ...).

The ``MoleculeLoader`` class itself is tested in ``pyaptamer/data/tests/``;
here we only check that the dataset ``load_*`` functions return a usable
loader.
"""

import pytest

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.datasets import load_1brq, load_1gnh, load_5nu7

# Molecule loaders that carry amino-acid sequences (have SEQRES records).
SEQUENCE_LOADERS = [load_1gnh, load_5nu7, load_1brq]


@pytest.mark.parametrize("loader", SEQUENCE_LOADERS)
def test_sequence_loader_materializes(loader):
    """Each protein loader returns a MoleculeLoader that materializes to data.

    Calling ``to_dataframe()`` actually exercises the loader -- a bare
    ``isinstance`` check would pass even on an unmigrated loader that errors
    on materialization.
    """
    mol = loader()
    assert isinstance(mol, MoleculeLoader)
    assert not mol.to_dataframe().empty


def test_load_1gnh_default_is_bag():
    """load_1gnh keeps the 10 chains of 1gnh in one list-valued cell."""
    df = load_1gnh().to_dataframe()

    assert df.shape == (1, 1)
    chains = df.iloc[0, 0]
    assert len(chains) == 10
    assert chains[0].startswith("QTDMSRK")


def test_load_1gnh_samples():
    """load_1gnh(tiling='samples') gives one row per chain."""
    df = load_1gnh(tiling="samples").to_dataframe()

    assert df.shape == (10, 1)
    assert df.iloc[0, 0].startswith("QTDMSRK")
