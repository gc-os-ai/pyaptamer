import numpy as np
import pandas as pd
import pytest

from pyaptamer.utils._aptanet_utils import pairs_to_features

APTAMER_SEQ = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
PROTEIN_SEQ = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"


@pytest.mark.parametrize(
    ("aptamer_col", "protein_col"),
    [
        ("aptamer", "protein"),
        ("aptamer_sequence", "target_sequence"),
    ],
)
def test_pairs_to_features_accepts_supported_dataframe_schemas(
    aptamer_col, protein_col
):
    """Supported DataFrame schemas should produce the same feature matrix."""
    pairs = [(APTAMER_SEQ, PROTEIN_SEQ), (APTAMER_SEQ, PROTEIN_SEQ)]
    df = pd.DataFrame(
        {
            aptamer_col: [APTAMER_SEQ, APTAMER_SEQ],
            protein_col: [PROTEIN_SEQ, PROTEIN_SEQ],
        }
    )

    expected = pairs_to_features(pairs)
    actual = pairs_to_features(df)

    np.testing.assert_allclose(actual, expected)


def test_pairs_to_features_rejects_unknown_dataframe_schema():
    """Unsupported DataFrame schemas should fail with a helpful message."""
    df = pd.DataFrame(
        {
            "aptamer_seq": [APTAMER_SEQ],
            "protein_seq": [PROTEIN_SEQ],
        }
    )

    with pytest.raises(ValueError, match="DataFrame input must contain"):
        pairs_to_features(df)
