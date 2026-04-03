import io
from unittest.mock import patch

import pandas as pd
import pytest
import requests
from Bio import SeqIO

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


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_pdbe_http_error(mock_get):
    """Test that PDBe API HTTP errors raise requests.HTTPError, not JSONDecodeError."""
    # Mock the first call (PDBe mapping) to return 503 with HTML body
    mock_response_503 = requests.Response()
    mock_response_503.status_code = 503
    mock_response_503._content = b"<html><body>Service Temporarily Unavailable</body></html>"
    mock_response_503.reason = "Service Unavailable"

    # Configure mock to return 503 for PDBe URL
    mock_get.return_value = mock_response_503

    with pytest.raises(requests.HTTPError) as excinfo:
        pdb_to_seq_uniprot("1a3n")

    assert "503" in str(excinfo.value)


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_pdbe_404(mock_get):
    """Test that PDBe API 404 raises HTTPError."""
    mock_response_404 = requests.Response()
    mock_response_404.status_code = 404
    mock_response_404._content = b"<html><body>Not Found</body></html>"
    mock_response_404.reason = "Not Found"

    mock_get.return_value = mock_response_404

    with pytest.raises(requests.HTTPError) as excinfo:
        pdb_to_seq_uniprot("nonexistent")

    assert "404" in str(excinfo.value)


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_uniprot_http_error(mock_get):
    """Test that UniProt API HTTP errors raise requests.HTTPError."""
    # First call (PDBe) returns valid mapping
    pdbe_response = requests.Response()
    pdbe_response.status_code = 200
    pdbe_response._content = b'{"1a3n": {"UniProt": {"P12345": {}}}}'

    # Second call (UniProt) returns 503
    uniprot_response = requests.Response()
    uniprot_response.status_code = 503
    uniprot_response._content = b"<html><body>Service Unavailable</body></html>"
    uniprot_response.reason = "Service Unavailable"

    mock_get.side_effect = [pdbe_response, uniprot_response]

    with pytest.raises(requests.HTTPError) as excinfo:
        pdb_to_seq_uniprot("1a3n")

    assert "503" in str(excinfo.value)


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_uniprot_404(mock_get):
    """Test that UniProt API 404 raises HTTPError."""
    pdbe_response = requests.Response()
    pdbe_response.status_code = 200
    pdbe_response._content = b'{"1a3n": {"UniProt": {"P12345": {}}}}'

    uniprot_response = requests.Response()
    uniprot_response.status_code = 404
    uniprot_response._content = b"<html><body>Not Found</body></html>"
    uniprot_response.reason = "Not Found"

    mock_get.side_effect = [pdbe_response, uniprot_response]

    with pytest.raises(requests.HTTPError) as excinfo:
        pdb_to_seq_uniprot("1a3n")

    assert "404" in str(excinfo.value)


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_success_list(mock_get):
    """Test successful retrieval with mocked responses, returning list."""
    pdbe_response = requests.Response()
    pdbe_response.status_code = 200
    pdbe_response._content = b'{"1a3n": {"UniProt": {"P12345": {"mappings": []}}}}'

    fasta_content = b">sp|P12345|Test protein\nMKTIIALSYIFCLVFADYKDDDKG"
    uniprot_response = requests.Response()
    uniprot_response.status_code = 200
    uniprot_response._content = fasta_content

    mock_get.side_effect = [pdbe_response, uniprot_response]

    result = pdb_to_seq_uniprot("1a3n", return_type="list")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == "MKTIIALSYIFCLVFADYKDDDKG"


@patch("pyaptamer.utils._pdb_to_seq_uniprot.requests.get")
def test_pdb_to_seq_uniprot_success_df(mock_get):
    """Test successful retrieval with mocked responses, returning DataFrame."""
    pdbe_response = requests.Response()
    pdbe_response.status_code = 200
    pdbe_response._content = b'{"1a3n": {"UniProt": {"P12345": {"mappings": []}}}}'

    fasta_content = b">sp|P12345|Test protein\nMKTIIALSYIFCLVFADYKDDDKG"
    uniprot_response = requests.Response()
    uniprot_response.status_code = 200
    uniprot_response._content = fasta_content

    mock_get.side_effect = [pdbe_response, uniprot_response]

    df = pdb_to_seq_uniprot("1a3n", return_type="pd.df")
    assert isinstance(df, pd.DataFrame)
    assert df["sequence"].iloc[0] == "MKTIIALSYIFCLVFADYKDDDKG"
