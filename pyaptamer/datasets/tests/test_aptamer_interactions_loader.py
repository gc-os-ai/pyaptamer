__author__ = "Satarupa22-SD"

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from pyaptamer.datasets import load_aptadb
from pyaptamer.datasets._loaders.load_aptamer_interactions import (
    load_interactions,
)


def test_local_csv(tmp_path):
    csv_path = tmp_path / "aptadb_sample.csv"
    pd.DataFrame({
        "aptamer_id": ["APT001"],
        "aptamer_sequence": ["AUGCUU"],
        "target_name": ["Thrombin"],
        "interaction_present": ["1"],
    }).to_csv(csv_path, index=False)

    df = load_interactions(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.loc[0, "aptamer_sequence"] == "AUGCUU"  # unchanged


def test_uses_cache(tmp_path):
    csv_path = tmp_path / "aptadb.csv"
    pd.DataFrame({"aptamer_sequence": ["AUGU"], "target_name": ["X"]}).to_csv(csv_path, index=False)

    with patch(
        "pyaptamer.datasets._loaders.load_aptamer_interactions.find_csv",
        return_value=csv_path,
    ):
        with patch(
            "pyaptamer.datasets._loaders.load_aptamer_interactions.download_dataset"
        ) as mock_dl:
            df = load_aptadb(cache_dir=tmp_path)
            assert not df.empty
            assert df.loc[0, "aptamer_sequence"] == "AUGU"
            mock_dl.assert_not_called()


def test_requires_kaggle(tmp_path):
    # Ensure no CSV present so a download would be attempted
    with patch.dict("sys.modules", {"kaggle": None}):
        with pytest.raises(ImportError):
            load_aptadb(cache_dir=tmp_path)


def test_invalid_dataset(tmp_path):
    # Force the download path and make it fail
    with patch(
        "pyaptamer.datasets._loaders.load_aptamer_interactions.find_csv",
        return_value=None,
    ):
        with patch(
            "pyaptamer.datasets._loaders.load_aptamer_interactions.download_dataset",
            side_effect=Exception("boom"),
        ):
            with pytest.raises(RuntimeError, match=r"Failed to download dataset .* from Kaggle"):
                load_aptadb("nonexistent/invalid-dataset", cache_dir=tmp_path)


@pytest.fixture
def sample_aptadb_data():
    return pd.DataFrame({
        'aptamer_id': ['APT001', 'APT002', 'APT003'],
        'target_id': ['TGT001', 'TGT002', 'TGT003'],
        'aptamer_sequence': ['ATCGATCGATCGATCG', 'GCTAGCTAGCTAGCTA', 'TTAACCGGTTAACCGG'],
        'target_name': ['Thrombin', 'VEGF', 'Lysozyme'],
        'target_uniprot': ['P00734', 'P15692', 'P61626'],
        'organism': ['Homo sapiens', 'Homo sapiens', 'Gallus gallus'],
        'ligand_type': ['Protein', 'Protein', 'Protein'],
        'binding_conditions': ['pH 7.4, 25°C', 'pH 7.0, 37°C', 'pH 8.0, 25°C'],
        'reference_pubmed_id': ['12345678', '87654321', '11223344'],
        'interaction_present': [1, 1, 0]
    })


def test_sample_columns(sample_aptadb_data):
    df = sample_aptadb_data
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    expected_columns = [
        'aptamer_id', 'target_id', 'aptamer_sequence', 'target_name',
        'target_uniprot', 'organism', 'ligand_type', 'binding_conditions',
        'reference_pubmed_id', 'interaction_present'
    ]

    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found in dataset"

    assert df['aptamer_sequence'].dtype == 'object'
    assert df['target_name'].dtype == 'object'


@pytest.mark.slow
def test_cache_consistency(tmp_path):
    # This test verifies that two consecutive calls yield same DataFrame when using cache.
    # It still avoids network by seeding the cache with a local CSV.
    csv_path = tmp_path / "aptadb.csv"
    seeded = pd.DataFrame({"aptamer_sequence": ["AU"], "target_name": ["X"], "interaction_present": [0]})
    seeded.to_csv(csv_path, index=False)

    with patch(
        "pyaptamer.datasets._loaders.load_aptamer_interactions.find_csv",
        return_value=csv_path,
    ):
        df1 = load_aptadb(cache_dir=tmp_path)
        df2 = load_aptadb(cache_dir=tmp_path)
        pd.testing.assert_frame_equal(df1, df2)