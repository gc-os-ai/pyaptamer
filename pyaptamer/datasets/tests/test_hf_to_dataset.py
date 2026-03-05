import os

from pyaptamer.datasets import load_hf_to_dataset


def test_hf_hub_dataset_load():
    """Test loading a known Hugging Face Hub dataset (small)."""
    ds = load_hf_to_dataset(
        "https://huggingface.co/datasets/gcos/HoloRBP4_round8_trimmed/resolve/main/HoloRBP4_round8_trimmed.fasta"
    )
    assert "text" in ds.column_names


def test_load_pdb_local_file():
    """Test parsing a local PDB file (pfoa.pdb) from the data folder."""
    pdb_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "pfoa.pdb"
    )
    ds = load_hf_to_dataset(pdb_file)
    assert "text" in ds.column_names


def test_download_locally_disallowed_host_raises():
    bad_url = "https://example.com/file.fasta"
    with pytest.raises(ValueError):
        load_hf_to_dataset(bad_url, download_locally=True)


def test_validate_url_allows_hf():
    # huggingface domain should still succeed (storage disabled for speed)
    url = "https://huggingface.co/datasets/gcos/HoloRBP4_round8_trimmed/resolve/main/HoloRBP4_round8_trimmed.fasta"
    # we don't actually download; just ensure no error is raised
    load_hf_to_dataset(url, download_locally=False)
