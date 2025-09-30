import os

from pyaptamer.utils.hf_to_dataset import hf_to_dataset


def test_hf_hub_dataset_load():
    """Test loading a known Hugging Face Hub dataset (small)."""
    ds = hf_to_dataset(
        "https://huggingface.co/datasets/gcos/HoloRBP4_round8_trimmed/resolve/main/HoloRBP4_round8_trimmed.fasta"
    )
    assert "text" in ds.column_names


def test_load_pdb_local_file():
    """Test parsing a local PDB file (pfoa.pdb) from the data folder."""
    pdb_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "data", "pfoa.pdb"
    )
    ds = hf_to_dataset(pdb_file)
    assert "text" in ds.column_names
