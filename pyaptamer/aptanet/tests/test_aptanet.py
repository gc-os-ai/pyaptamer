import numpy as np
import pytest
import torch
from sklearn.utils.estimator_checks import check_estimator

from pyaptamer.aptanet import FeatureSelector, SkorchAptaNet
from pyaptamer.aptanet.pipeline import pairs_to_features


@pytest.mark.parametrize(
    "aptamer_seq, protein_seq",
    [
        (
            "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC",
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
        )
    ],
)
def test_model_fit_and_predict(aptamer_seq, protein_seq):
    """
    Test SkorchAptaNet fitting and predicting on synthetic data.

    Asserts
    -------
    Model predictions are valid class labels and shape matches input.
    """
    # Generate dummy input
    X_raw = [(aptamer_seq, protein_seq) for _ in range(40)]
    y = np.array([0] * 20 + [1] * 20, dtype=np.float32)

    # Convert sequence pairs to features
    X = pairs_to_features(X_raw)

    # Initialize model
    net = SkorchAptaNet(
        module__hidden_dim=64,
        module__n_hidden=3,
        module__dropout=0.2,
        max_epochs=5,
        batch_size=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0,
    )

    net.fit(X, y)
    preds = net.predict(X)

    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize("estimator", [FeatureSelector()])
def test_sklearn_compatible_estimator(estimator):
    """
    Test that FeatureSelector is compatible with scikit-learn estimator checks.
    """
    check_estimator(estimator)
