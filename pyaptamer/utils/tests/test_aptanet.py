import numpy as np
import pandas as pd
from pyaptamer.utils._aptanet_utils import generate_kmer_vecs
from pyaptamer.utils._aptanet_utils import pairs_to_features


def test_generate_kmer_vecs_shape():
    seq = "ACGT"
    k = 2
    vec = generate_kmer_vecs(seq, k=k)

    expected_size = sum(4 ** i for i in range(1, k + 1))
    assert vec.shape == (expected_size,)


def test_generate_kmer_vecs_normalization():
    seq = "AAAA"
    vec = generate_kmer_vecs(seq, k=1)

    assert np.isclose(np.sum(vec), 1.0)
    assert vec[0] == 1.0


def test_pairs_to_features_dataframe():
    long_protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLM"
    df = pd.DataFrame({
        "aptamer": ["ACGT", "AAAA"],
        "protein": [long_protein, long_protein]
    })
    feats = pairs_to_features(df, k=2)
    assert isinstance(feats, np.ndarray)
    assert feats.shape[0] == 2


def test_pairs_to_features_list():
    """Check that passing a list of tuples works correctly."""
    long_protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLM"
    data = [("ACGT", long_protein), ("AAAA", long_protein)]
    feats = pairs_to_features(data, k=2)
    assert isinstance(feats, np.ndarray)
    assert feats.shape[0] == 2