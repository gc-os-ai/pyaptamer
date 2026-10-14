"""End-to-end tests for AptamerEvalAptaMCTS with a fitted pipeline and MCTS."""

__author__ = ["siddharth7113"]

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_li2014
from pyaptamer.experiments import AptamerEvalAptaMCTS
from pyaptamer.mcts import MCTS


@pytest.fixture(scope="module")
def fitted():
    """A pipeline fitted on the li2014 test split and one target protein."""
    X, y = load_li2014(split="test", return_X_y=True)
    estimator = RandomForestClassifier(n_estimators=10, random_state=0)
    pipeline = AptaMCTSPipeline(estimator=estimator).fit(X, y["label"].to_numpy())
    target = X.to_dataframe()["protein"].iloc[0]
    return pipeline, target


def test_evaluate_is_the_positive_class_probability(fitted):
    """evaluate returns predict_proba of the positive class for the pair."""
    pipeline, target = fitted
    experiment = AptamerEvalAptaMCTS(target=target, pipeline=pipeline)
    pair = MoleculeLoader(data={"aptamer": ["ACGUACGUAC"], "protein": [target]})
    score = experiment.evaluate("ACGUACGUAC")
    assert isinstance(score, np.float64)
    assert score == pipeline.predict_proba(pair)[0, 1]


def test_mcts_runs_with_the_experiment(fitted):
    """MCTS driven by the experiment returns a candidate and a score in [0, 1]."""
    pipeline, target = fitted
    experiment = AptamerEvalAptaMCTS(target=target, pipeline=pipeline)
    result = MCTS(experiment=experiment, depth=3, n_iterations=5).run(verbose=False)
    assert isinstance(result["candidate"], str)
    assert len(result["candidate"]) > 0
    assert 0.0 <= result["score"] <= 1.0


@pytest.mark.parametrize("candidate", ["", "ACGU_"])
def test_evaluate_handles_empty_and_unknown_letters(fitted, candidate):
    """An empty candidate or one with a letter outside ACGU still gets a score."""
    pipeline, target = fitted
    experiment = AptamerEvalAptaMCTS(target=target, pipeline=pipeline)
    score = experiment.evaluate(candidate)
    assert 0.0 <= score <= 1.0


def test_evaluate_uses_the_pipeline_column_names():
    """A pipeline fitted on renamed columns scores pairs built with those names."""
    X, y = load_li2014(split="test", return_X_y=True)
    frame = X.to_dataframe().rename(columns={"aptamer": "apt", "protein": "target"})
    estimator = RandomForestClassifier(n_estimators=10, random_state=0)
    pipeline = AptaMCTSPipeline(
        aptamer_col="apt", protein_col="target", estimator=estimator
    ).fit(frame, y["label"].to_numpy())
    target = frame["target"].iloc[0]
    experiment = AptamerEvalAptaMCTS(target=target, pipeline=pipeline)
    pair = MoleculeLoader(data={"apt": ["ACGUACGUAC"], "target": [target]})
    assert experiment.evaluate("ACGUACGUAC") == pipeline.predict_proba(pair)[0, 1]
