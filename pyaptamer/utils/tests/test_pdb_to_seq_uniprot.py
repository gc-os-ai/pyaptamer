import pandas as pd
import requests

from pyaptamer.utils import pdb_to_seq_uniprot


def test_pdb_to_seq_uniprot(monkeypatch):
    """Test the `pdb_to_seq_uniprot` function with mocked network calls."""
    pdb_id = "1a3n"

    class FakeResp:
        def __init__(self, json_data=None, text_data=None, status=200):
            self._json = json_data
            self._text = text_data
            self.status_code = status

        @property
        def ok(self):
            return 200 <= self.status_code < 300

        def json(self):
            return self._json

        @property
        def text(self):
            return self._text


    def fake_get(url, *args, **kwargs):
        if "pdbe/api/mappings/uniprot" in url:
            # return a mapping with a UniProt id
            return FakeResp(json_data={"1a3n": {"UniProt": {"P00734": {}}}})
        if "rest.uniprot.org/uniprotkb" in url:
            # return a minimal FASTA
            return FakeResp(text_data=">P00734\nMKT\n")
        return FakeResp(status=404)

    monkeypatch.setattr(requests, "get", fake_get)

    df = pdb_to_seq_uniprot(pdb_id, return_type="pd.df")
    assert isinstance(df, pd.DataFrame)
    assert "sequence" in df.columns
    assert len(df.iloc[0]["sequence"]) > 0

    lst = pdb_to_seq_uniprot(pdb_id, return_type="list")
    assert isinstance(lst, list)
    assert len(lst) == 1
    assert len(lst[0]) > 0
