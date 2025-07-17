import numpy as np
import pytest

from pyaptamer.aptanet import AptaNet
from pyaptamer.pseaac import PSeAAC


@pytest.mark.parametrize(
    "aptamer_seq, protein_seq",
    [
        (
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
            "AGCTTAGCGTACAGCTTAGCGTAC",
        )
    ],
)
def test_model_fit_and_predict(aptamer_seq, protein_seq):
    """
    Test AptaNet fitting and predicting on synthetic data.

    Asserts
    -------
    Model predictions are valid class labels and shape matches input.
    """
    model = AptaNet(n_layers=3, hidden_dim=64, dropout=0.2)

    # Generate dummy feature matrix
    X = np.array([model.preprocessing(aptamer_seq, protein_seq) for _ in range(40)])
    y = np.array([0] * 20 + [1] * 20)

    model.fit(X, y, epochs=5, batch_size=10)

    preds = model.predict(X)
    assert preds.shape == (40,)
    assert set(preds).issubset({0, 1})


def test_invalid_protein_sequence():
    """
    Test PSeAAC raises ValueError on invalid amino acids.

    Asserts
    -------
    ValueError is raised for an invalid protein sequence.
    """
    pseaac = PSeAAC()
    with pytest.raises(ValueError):
        pseaac.transform("ACDEXFGHIKL")  # 'X' is not a valid amino acid
