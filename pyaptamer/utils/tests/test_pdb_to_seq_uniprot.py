import pandas as pd

from pyaptamer.utils import pdb_to_seq_uniprot


def test_pdb_to_seq_uniprot():
    """Test the `pdb_to_seq_uniprot` function."""
    pdb_id = "1a3n"

    df = pdb_to_seq_uniprot(pdb_id, return_type="pd.df")
    assert isinstance(df, pd.DataFrame)
    assert "sequence" in df.columns
    assert len(df.iloc[0]["sequence"]) > 0

    lst = pdb_to_seq_uniprot(pdb_id, return_type="list")
    assert isinstance(lst, list)
    assert len(lst) == 1
    assert len(lst[0]) > 0
