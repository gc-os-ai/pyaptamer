import os

import pytest
import requests

from pyaptamer.datasets import load_hf_to_dataset
from pyaptamer.datasets._loaders import _hf_to_dataset_loader

FASTA_URL = "https://example.invalid/datasets/dummy.fasta"
FASTA_BYTES = b">seq1\nACGT\n"


class _StubResponse:
    """Stand-in for a ``requests`` response, exposing only what the loader uses."""

    def __init__(self, content=FASTA_BYTES, error=None):
        self.content = content
        self._error = error

    def raise_for_status(self):
        """Raise the configured error, like a real response would on a bad status."""
        if self._error is not None:
            raise self._error


def _patch_requests_get(monkeypatch, calls, error=None):
    """Replace ``requests.get`` in the loader with a stub recording every call."""

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        return _StubResponse(error=error)

    monkeypatch.setattr(_hf_to_dataset_loader.requests, "get", fake_get)


def _patch_load_dataset(monkeypatch, calls, returns):
    """Replace ``load_dataset`` in the loader so nothing reaches the HF Hub."""

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return returns

    monkeypatch.setattr(_hf_to_dataset_loader, "load_dataset", fake_load_dataset)


def test_hf_hub_dataset_load():
    """Test loading a known Hugging Face Hub dataset (small)."""
    ds = load_hf_to_dataset(
        "https://huggingface.co/datasets/gcos/HoloRBP4_round8_trimmed/resolve/main/HoloRBP4_round8_trimmed.fasta"
    )
    assert "text" in ds.column_names


def test_load_pdb_local_file():
    """Test parsing a local PDB file from the data folder."""
    pdb_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "1brq.pdb"
    )
    ds = load_hf_to_dataset(pdb_file)
    assert "text" in ds.column_names


def test_download_locally_writes_file_and_loads_it(tmp_path, monkeypatch):
    """download_locally=True saves the URL under ./hf_datasets/ and loads that copy."""
    monkeypatch.chdir(tmp_path)
    get_calls = []
    load_calls = []
    _patch_requests_get(monkeypatch, get_calls)
    _patch_load_dataset(monkeypatch, load_calls, {"train": "a", "test": "b"})

    ds = load_hf_to_dataset(FASTA_URL, download_locally=True)

    local_path = tmp_path / "hf_datasets" / "dummy.fasta"
    assert local_path.read_bytes() == FASTA_BYTES
    assert get_calls == [FASTA_URL]
    assert load_calls[0][0] == ("text",)
    assert load_calls[0][1]["data_files"] == os.path.join("hf_datasets", "dummy.fasta")
    assert ds == {"train": "a", "test": "b"}


def test_download_locally_reuses_cached_file(tmp_path, monkeypatch):
    """A second load of the same URL reads the cached file instead of downloading."""
    monkeypatch.chdir(tmp_path)
    get_calls = []
    load_calls = []
    _patch_requests_get(monkeypatch, get_calls)
    _patch_load_dataset(monkeypatch, load_calls, {"train": "a", "test": "b"})

    load_hf_to_dataset(FASTA_URL, download_locally=True)
    load_hf_to_dataset(FASTA_URL, download_locally=True)

    assert get_calls == [FASTA_URL]
    assert len(load_calls) == 2
    assert (tmp_path / "hf_datasets" / "dummy.fasta").read_bytes() == FASTA_BYTES


def test_download_locally_propagates_http_error(tmp_path, monkeypatch):
    """A failing download raises and leaves no partial file behind."""
    monkeypatch.chdir(tmp_path)
    get_calls = []
    load_calls = []
    _patch_requests_get(monkeypatch, get_calls, error=requests.HTTPError("404"))
    _patch_load_dataset(monkeypatch, load_calls, {"train": "a"})

    with pytest.raises(requests.HTTPError):
        load_hf_to_dataset(FASTA_URL, download_locally=True)

    assert not (tmp_path / "hf_datasets" / "dummy.fasta").exists()
    assert load_calls == []


def test_single_split_dict_is_unwrapped(monkeypatch):
    """A dataset with one split is returned directly rather than as a mapping."""
    load_calls = []
    _patch_load_dataset(monkeypatch, load_calls, {"train": "only-split"})

    ds = load_hf_to_dataset("some-org/some-dataset")

    assert ds == "only-split"
    assert load_calls[0][0] == ("some-org/some-dataset",)
