import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

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
    if json_data is not None:
        mock.json.return_value = json_data
    if text is not None:
        mock.text = text
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock
        )
    else:
        mock.raise_for_status.return_value = None
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


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_pdbe_503_raises_http_error(mock_get):
    """Check that an HTTP 503 from PDBe raises HTTPError instead of JSONDecodeError."""
    mock_get.return_value = _make_mock_response(status_code=503)

    with pytest.raises(requests.exceptions.HTTPError):
        pdb_to_seq_uniprot("1a3n")


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_uniprot_error_raises_http_error(mock_get):
    """Check that an HTTP error from UniProt raises HTTPError."""
    mock_get.side_effect = [
        _make_mock_response(json_data=_MAPPING_JSON),
        _make_mock_response(status_code=404),
    ]

    with pytest.raises(requests.exceptions.HTTPError):
        pdb_to_seq_uniprot("1a3n")


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_no_uniprot_mapping_raises_value_error(mock_get):
    """Check that missing UniProt mapping raises ValueError."""
    mock_get.return_value = _make_mock_response(json_data={"1a3n": {"UniProt": {}}})

    with pytest.raises(ValueError, match="No UniProt mapping found"):
        pdb_to_seq_uniprot("1a3n")


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_invalid_return_type(mock_get):
    """Check that an invalid return_type raises ValueError."""
    mock_get.side_effect = [
        _make_mock_response(json_data=_MAPPING_JSON),
        _make_mock_response(text=_FASTA_TEXT),
    ]

    with pytest.raises(ValueError, match="return_type"):
        pdb_to_seq_uniprot("1a3n", return_type="invalid")
