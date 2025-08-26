__author__ = "satvshr"

import pytest

from pyaptamer.deepatamer._model import DeepAptamerNN
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
    model = DeepAptamerNN()
    pipe = DeepAptamerPipeline(model=model, device="cpu")

    ranked = pipe.predict(seqs)

    # Ensure sorted in descending order
    scores = [item["score"] for item in ranked]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
