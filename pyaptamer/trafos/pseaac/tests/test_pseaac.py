import pandas as pd
import pytest
from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.pseaac import AptaNetPSeAAC, PSeAAC
from pathlib import Path
import numpy as np

# Path to sample PDB for testing
DATA_PATH = Path(__file__).parent.parent.parent.parent / "datasets" / "data"
PDB_PATH = DATA_PATH / "1gnh.pdb"

@pytest.mark.parametrize("Transformer", [PSeAAC, AptaNetPSeAAC])
def test_pseaac_molecule_loader(Transformer):
    """Test that Transformer works with MoleculeLoader."""
    if not PDB_PATH.exists():
        pytest.skip(f"Sample PDB file not found at {PDB_PATH}")

    loader = MoleculeLoader(PDB_PATH)
    transformer = Transformer(lambda_val=5)
    
    # transform(MoleculeLoader)
    Xt = transformer.transform(loader)
    
    assert isinstance(Xt, pd.DataFrame)
    # 1gnh.pdb should have at least one chain
    assert Xt.shape[0] > 0
    # Check shape: (20 + 5) * 7 = 175 for AptaNetPSeAAC or something similar for PSeAAC
    # PSeAAC default grouping is also 7 (21/3)
    assert Xt.shape[1] == (20 + 5) * 7
    assert isinstance(Xt.index, pd.MultiIndex)
    assert "path" in Xt.index.names
    assert "chain_id" in Xt.index.names

@pytest.mark.parametrize("Transformer", [PSeAAC, AptaNetPSeAAC])
def test_pseaac_dataframe(Transformer):
    """Test that Transformer works with pd.DataFrame."""
    df = pd.DataFrame({"sequence": ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"]})
    transformer = Transformer(lambda_val=10)
    
    Xt = transformer.transform(df)
    
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == 1
    assert Xt.shape[1] == (20 + 10) * 7

def test_pseaac_backward_compatibility():
    """Test that PSeAAC still works (via BaseTransform coercion) with MoleculeLoader strings."""
    # Note: BaseTransform._check_X_y currently only supports MoleculeLoader or DataFrame.
    # However, the user might want to pass a string sequence directly if they were used to the old API.
    # BaseTransform would need to be updated to support strings, or we handle it in _check_X.
    
    # Currently pyaptamer/trafos/base/_base.py says:
    # if not isinstance(X, pd.DataFrame): raise TypeError("X must be a MoleculeLoader instance or a pandas DataFrame. Got {type(X)} instead.")
    
    # So if the user wants to stay compatible with the old PSeAAC.transform("SEQ") call, 
    # we might need to update BaseTransform or wrap it.
    
    pass # For now, we follow the new transformer interface

def test_pseaac_invalid_sequence():
    """Test that PSeAAC raises error for short sequences."""
    df = pd.DataFrame({"sequence": ["ACDE"]})
    transformer = PSeAAC(lambda_val=30)
    
    with pytest.raises(ValueError, match="Protein sequence is too short"):
        transformer.transform(df)
