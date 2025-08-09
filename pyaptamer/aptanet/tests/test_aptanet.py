import sys

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pyaptamer.aptanet import AptaPipeline


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="skorch does not support Python 3.13"
)
@pytest.mark.parametrize(
    "aptamer_seq, protein_seq",
    [
        (
            "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC",
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
        )
    ],
)
def test_pipeline_fit_and_predict(aptamer_seq, protein_seq):
    """
    Test if Pipeline predictions are valid class labels and shape matches input.
    """
    pipe = AptaPipeline()

    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    pipe.fit(X_raw, y)
    preds = pipe.predict(X_raw)

    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="skorch does not support Python 3.13"
)
@parametrize_with_checks([AptaPipeline()])
def test_sklearn_compatible_estimator(estimator, check):
    """
    Run scikit-learn's compatibility checks on the AptaPipeline.
    """
    check(estimator)
