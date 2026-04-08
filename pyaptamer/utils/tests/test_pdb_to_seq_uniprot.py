from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pyaptamer.utils import pdb_to_seq_uniprot

_MAPPING_JSON = {
    "1a3n": {
        "UniProt": {
            "P69905": {},
        }
    }
}

_FASTA_TEXT = (
    ">sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens\n"
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG\n"
    "KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP\n"
    "AVHASLDKFLASVSTVLTSKYR\n"
)


def _make_mock_response(json_data=None, text=None, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.raise_for_status.return_value = None
    if json_data is not None:
        mock.json.return_value = json_data
    if text is not None:
        mock.text = text
    return mock


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_returns_list(mock_get):
    """Check pdb_to_seq_uniprot returns a list with return_type='list'."""
    mock_get.side_effect = [
        _make_mock_response(json_data=_MAPPING_JSON),
        _make_mock_response(text=_FASTA_TEXT),
    ]

    lst = pdb_to_seq_uniprot("1a3n", return_type="list")

    assert isinstance(lst, list)
    assert len(lst) == 1
    assert len(lst[0]) > 0


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_returns_dataframe(mock_get):
    """Check pdb_to_seq_uniprot returns a DataFrame with return_type='pd.df'."""
    mock_get.side_effect = [
        _make_mock_response(json_data=_MAPPING_JSON),
        _make_mock_response(text=_FASTA_TEXT),
    ]

    df = pdb_to_seq_uniprot("1a3n", return_type="pd.df")

    assert isinstance(df, pd.DataFrame)
    assert "sequence" in df.columns
    assert len(df.iloc[0]["sequence"]) > 0


def test_pdb_to_seq_uniprot_no_mapping():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"1abc": {}}
        with pytest.raises(ValueError):
            pdb_to_seq_uniprot("1abc")

def test_pdb_to_seq_uniprot_invalid_return():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"1a3n": {"UniProt": {"P12345": {}}}}
        mock_get.return_value.text = ">test\nVLSP"
        with pytest.raises(ValueError):
            pdb_to_seq_uniprot("1a3n", return_type="json")