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


def test_invalid_pdb_raises():
    with pytest.raises(ValueError):
        pdb_to_seq_uniprot("bad!")


def test_nonexistent_pdb_raises(monkeypatch):
    def fake_get(url):
        class R:
            def json(self):
                return {}
        return R()

    monkeypatch.setattr("requests.get", fake_get)
    with pytest.raises(ValueError):
        pdb_to_seq_uniprot("1abc")
