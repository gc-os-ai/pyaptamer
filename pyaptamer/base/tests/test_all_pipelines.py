"""Test collection for all BasePipelineDelegator estimators in pyaptamer.

skbase's ``BaseFixtureGenerator`` finds the pipelines and builds one test
instance per ``get_test_params`` entry; ``TestAllObjects`` adds the skbase
tests. Input frames come from the ``col_params`` tag: ``SEQUENCES`` holds
one list of sequences per column parameter name, so a new pipeline that
uses the same parameter names needs no test data of its own.
"""

__author__ = ["siddharth7113"]

import numpy as np
import pandas as pd
import pytest
from skbase._exceptions import NotFittedError
from skbase.testing import BaseFixtureGenerator, TestAllObjects
from sklearn.base import is_classifier, is_regressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from pyaptamer.base import BasePipelineDelegator
from pyaptamer.data import MoleculeLoader

SEQUENCES = {
    "aptamer_col": [
        "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC",
        "TGCATGCTAGCTAGCTAGCTAGCTAGCTAGCGCTA",
    ],
    "protein_col": [
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV",
    ],
}
N_ROWS = 8


def _frame(obj, n=N_ROWS):
    """A frame with obj's configured columns, the two sequences alternating."""
    return pd.DataFrame(
        {getattr(obj, p): (SEQUENCES[p] * n)[:n] for p in obj.get_tag("col_params")}
    )


def _labels(n=N_ROWS):
    return np.array([0, 1] * (n // 2))


class PackageConfig:
    """Config that the skbase test classes read."""

    package_name = "pyaptamer"

    valid_tags = ["object_type", "authors", "maintainers", "col_params"]


class PipelineFixtureGenerator(PackageConfig, BaseFixtureGenerator):
    """Creates the object_class and object_instance test arguments.

    Classes come from searching pyaptamer for BasePipelineDelegator
    subclasses. Instances are built fresh for every test, one per
    get_test_params entry.
    """

    object_type_filter = BasePipelineDelegator


class TestAllPipelines(PipelineFixtureGenerator, TestAllObjects):
    """Tests every BasePipelineDelegator subclass must pass.

    The skbase tests come in through ``TestAllObjects``. The tests below
    check what ``BasePipelineDelegator`` promises: input coercion, column
    selection, the fitted flag, and the scikit-learn classifier contract.
    """

    def test_column_params_have_test_sequences(self, object_class):
        """Every column parameter of the class has sequences in SEQUENCES.

        A parameter without sequences would leave the tests below with no
        input data and a KeyError instead of a clear message.
        """
        assert set(object_class.get_class_tag("col_params")) <= set(SEQUENCES)

    def test_fit_predict_and_predict_proba(self, object_instance):
        """fit returns self; predict and predict_proba have one row per sample."""
        X, y = _frame(object_instance), _labels()
        assert object_instance.fit(X, y) is object_instance
        assert object_instance.is_fitted is True
        assert object_instance.predict(X).shape == (N_ROWS,)
        proba = object_instance.predict_proba(X)
        assert proba.shape == (N_ROWS, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_array_equal(object_instance.classes_, [0, 1])

    @pytest.mark.parametrize("method", ["predict", "predict_proba", "score"])
    def test_methods_raise_before_fit(self, object_instance, method):
        """predict, predict_proba and score on an unfitted pipeline raise."""
        X, y = _frame(object_instance), _labels()
        args = (X, y) if method == "score" else (X,)
        with pytest.raises(NotFittedError):
            getattr(object_instance, method)(*args)

    def test_score_is_accuracy_of_predict(self, object_instance):
        """score with a classifier is the accuracy of predict."""
        X, y = _frame(object_instance), _labels()
        object_instance.fit(X, y)
        expected = accuracy_score(y, object_instance.predict(X))
        assert object_instance.score(X, y) == expected

    def test_moleculeloader_matches_dataframe(self, object_instance):
        """A MoleculeLoader over a frame predicts the same as the frame."""
        X, y = _frame(object_instance), _labels()
        object_instance.fit(MoleculeLoader(data=X), y)
        np.testing.assert_array_equal(
            object_instance.predict_proba(MoleculeLoader(data=X)),
            object_instance.predict_proba(X),
        )

    def test_rejects_other_input_types(self, object_instance):
        """Input that is neither a MoleculeLoader nor a DataFrame is a TypeError."""
        X = _frame(object_instance)
        with pytest.raises(TypeError, match="MoleculeLoader"):
            object_instance.fit(X.to_numpy().tolist(), _labels())

    def test_missing_column_names_class_and_parameter(self, object_instance):
        """A frame without a configured column raises and names the class and parameter."""  # noqa: E501
        X = _frame(object_instance)
        param = object_instance.get_tag("col_params")[0]
        X = X.rename(columns={getattr(object_instance, param): "other"})
        with pytest.raises(ValueError, match=type(object_instance).__name__) as err:
            object_instance.fit(X, _labels())
        assert param in str(err.value)

    def test_column_params_select_the_columns(self, object_instance):
        """The column parameters pick the columns, whatever they are called."""
        renamed = {p: f"my_{p}" for p in object_instance.get_tag("col_params")}
        object_instance.set_params(**renamed)
        X, y = _frame(object_instance), _labels()
        assert list(X.columns) == list(renamed.values())
        assert object_instance.fit(X, y).predict(X).shape == (N_ROWS,)

    def test_other_columns_do_not_reach_the_pipeline(self, object_instance):
        """Columns not named by the column parameters are dropped before fit."""
        X, y = _frame(object_instance), _labels()
        X = X.assign(note=["x"] * N_ROWS, round=[3] * N_ROWS)
        assert object_instance.fit(X, y).predict(X).shape == (N_ROWS,)

    @pytest.mark.parametrize("scoring", [None, "roc_auc"])
    def test_cross_val_score(self, object_instance, scoring):
        """cross_val_score gives finite scores, with the default score and roc_auc."""
        X, y = _frame(object_instance), _labels()
        scores = cross_val_score(object_instance, X, y, cv=2, scoring=scoring)
        assert np.isfinite(scores).all()

    def test_estimator_type_follows_the_estimator(self, object_instance):
        """scikit-learn sees the type of the estimator parameter."""
        if "estimator" not in object_instance.get_params():
            pytest.skip("the pipeline has no estimator parameter")
        assert is_classifier(object_instance)
        object_instance.set_params(estimator=DummyRegressor())
        assert is_regressor(object_instance)
