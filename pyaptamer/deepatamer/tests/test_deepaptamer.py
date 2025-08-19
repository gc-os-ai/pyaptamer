import pytest
import torch

from pyaptamer.deepatamer._model import DeepAptamer
from pyaptamer.deepatamer._pipeline import DeepAptamerPipeline


@pytest.mark.parametrize(
    "seqs",
    [
        "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTCCG",
        [
            "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC",
            "TGCATGCTAGCTAGCTAGCTAGCTAGCTAGCGCTA",
        ],
    ],
)
def test_pipeline_predict_shapes(seqs):
    """
    Test if DeepAptamerPipeline outputs valid ranked predictions.

    Raises
    ------
    AssertionError
        If prediction scores are not sorted in descending order.
    """
    model = DeepAptamer()
    pipe = DeepAptamerPipeline(model=model, device="cpu")

    ranked = pipe.predict(seqs)

    # Ensure sorted in descending order
    scores = [item["score"] for item in ranked]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_train_loop_runs():
    """
    This test creates synthetic data and ensures that the training loop runs without
    errors for a single epoch.

    Raises
    ------
    RuntimeError
        If there are issues in the forward or backward pass.
    """
    X_ohe = torch.rand(8, 35, 4)
    X_shape = torch.rand(8, 1, 126)
    y = torch.randint(0, 2, (8,))

    model = DeepAptamer()

    model.train_loop(X_ohe, X_shape, y, epochs=1, device="cpu")
