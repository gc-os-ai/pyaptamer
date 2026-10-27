"""scikit-learn model selection on the end-to-end classifier pipelines."""

__author__ = ["siddharth7113"]

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_regressor
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.aptanet import AptaNetPipeline, AptaNetRegressor
from pyaptamer.deepaptamer import DeepAptamerPipeline

APTAMER = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC"
PROTEIN = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
PAIRS = pd.DataFrame({"aptamer": [APTAMER] * 8, "protein": [PROTEIN] * 8})
LABELS = np.array([0, 1] * 4)

PIPELINES = [
    AptaMCTSPipeline(estimator=DummyClassifier()),
    AptaNetPipeline(estimator=DummyClassifier()),
    DeepAptamerPipeline(estimator=DummyClassifier()),
]


@pytest.mark.parametrize("scoring", [None, "roc_auc"])
@pytest.mark.parametrize("pipeline", PIPELINES, ids=lambda p: type(p).__name__)
def test_cross_val_score_gives_finite_scores(pipeline, scoring):
    """cross_val_score gives finite scores, with the default score and roc_auc."""
    scores = cross_val_score(pipeline, PAIRS, LABELS, cv=2, scoring=scoring)
    assert np.isfinite(scores).all()


def test_aptanet_pipeline_is_regressor_with_regressor_estimator():
    """AptaNetPipeline takes its estimator type from its estimator."""
    assert is_regressor(AptaNetPipeline(estimator=AptaNetRegressor()))
