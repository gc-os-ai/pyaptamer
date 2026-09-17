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


def test_load_1gnh_to_dataframe():
    """load_1gnh materializes to one row per chain (tiling='samples')."""
    df = load_1gnh().to_dataframe()

    # 1gnh has 10 chains -> 10 rows, single sequence column
    assert df.shape == (10, 1)
    assert df.iloc[0, 0].startswith("QTDMSRK")
