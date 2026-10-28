"""Tests for the AptaMCTS pipeline on the li2014 benchmark data."""

__author__ = ["aditi-dsi", "siddharth7113"]

import numpy as np
import pytest
from skbase._exceptions import NotFittedError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from pyaptamer.aptamcts import AptaMCTSPipeline
from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_li2014


@pytest.fixture(scope="module")
def li2014():
    """The li2014 test split as a MoleculeLoader and a label array."""
    X, y = load_li2014(split="test", return_X_y=True)
    return X, y["label"].to_numpy()


def _fast_rf():
    return RandomForestClassifier(n_estimators=10, random_state=0)


def test_predict_returns_one_label_per_row(li2014):
    """predict gives one of the training labels for every row."""
    X, y = li2014
    preds = AptaMCTSPipeline().fit(X, y).predict(X)
    assert preds.shape == (len(y),)
    assert set(preds) <= {"negative", "positive"}


def test_predict_proba_rows_sum_to_one(li2014):
    """predict_proba gives two columns per row that sum to one."""
    X, y = li2014
    proba = AptaMCTSPipeline(estimator=_fast_rf()).fit(X, y).predict_proba(X)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_default_feature_width_is_ictf(li2014):
    """Default k values give 340 aptamer and 399 protein features."""
    X, y = li2014
    pipe = AptaMCTSPipeline(estimator=_fast_rf()).fit(X, y)
    assert pipe.pipeline_[:-1].transform(X.to_dataframe()).shape[1] == 340 + 399


def test_k_values_set_feature_width(li2014):
    """rna_k and prot_k change the number of features."""
    X, y = li2014
    pipe = AptaMCTSPipeline(rna_k=1, prot_k=1, estimator=_fast_rf()).fit(X, y)
    assert pipe.pipeline_[:-1].transform(X.to_dataframe()).shape[1] == 4 + 7


def test_dataframe_and_loader_give_same_result(li2014):
    """A DataFrame and a MoleculeLoader over it predict the same."""
    X, y = li2014
    pipe = AptaMCTSPipeline(estimator=_fast_rf()).fit(X, y)
    np.testing.assert_array_equal(
        pipe.predict_proba(X), pipe.predict_proba(X.to_dataframe())
    )


def test_dna_and_rna_spelling_predict_the_same(li2014):
    """T and U spellings of one aptamer give the same probability."""
    X, y = li2014
    pipe = AptaMCTSPipeline(estimator=_fast_rf()).fit(X, y)
    protein = X.to_dataframe()["protein"].iloc[0]
    dna = MoleculeLoader(data={"aptamer": ["ACGTACGTAC"], "protein": [protein]})
    rna = MoleculeLoader(data={"aptamer": ["acguacguac"], "protein": [protein]})
    np.testing.assert_array_equal(pipe.predict_proba(dna), pipe.predict_proba(rna))


def test_custom_column_names(li2014):
    """aptamer_col and protein_col select the columns to encode."""
    X, y = li2014
    df = X.to_dataframe().rename(columns={"aptamer": "apt", "protein": "target"})
    pipe = AptaMCTSPipeline(
        aptamer_col="apt", protein_col="target", estimator=_fast_rf()
    )
    assert pipe.fit(df, y).predict(df).shape == (len(y),)


def test_swapped_estimator(li2014):
    """Any sklearn classifier can replace the random forest."""
    X, y = li2014
    pipe = AptaMCTSPipeline(estimator=LogisticRegression(max_iter=1000)).fit(X, y)
    assert isinstance(pipe.pipeline_[-1], LogisticRegression)
    assert pipe.predict_proba(X).shape == (len(y), 2)


def test_passed_estimator_is_not_fitted_in_place(li2014):
    """fit works on a clone, so the estimator passed in stays unfitted."""
    X, y = li2014
    rf = _fast_rf()
    AptaMCTSPipeline(estimator=rf).fit(X, y)
    assert not hasattr(rf, "estimators_")


@pytest.mark.parametrize("scoring", [None, "roc_auc"])
def test_cross_val_score(li2014, scoring):
    """cross_val_score gives finite scores, with the default score and roc_auc."""
    X, y = li2014
    pipe = AptaMCTSPipeline(estimator=_fast_rf())
    assert np.isfinite(
        cross_val_score(pipe, X.to_dataframe(), y, cv=2, scoring=scoring)
    ).all()


def test_predict_before_fit_raises(li2014):
    """predict on an unfitted pipeline raises NotFittedError."""
    X, _ = li2014
    with pytest.raises(NotFittedError):
        AptaMCTSPipeline().predict(X)


def test_rejects_other_input_types(li2014):
    """Input that is neither a MoleculeLoader nor a DataFrame is a TypeError."""
    _, y = li2014
    with pytest.raises(TypeError, match="MoleculeLoader"):
        AptaMCTSPipeline().fit([["ACGU", "MKT"]], y[:1])


def test_missing_column_raises(li2014):
    """A frame without the named columns is a ValueError naming them."""
    X, y = li2014
    df = X.to_dataframe().rename(columns={"protein": "target"})
    with pytest.raises(ValueError, match="protein"):
        AptaMCTSPipeline().fit(df, y)


def test_extra_columns_are_ignored(li2014):
    """Columns other than the aptamer and protein columns do not reach the estimator."""
    X, y = li2014
    df = X.to_dataframe().assign(label=y, note="x")
    pipe = AptaMCTSPipeline(estimator=_fast_rf()).fit(df, y)
    assert pipe.predict(df).shape == (len(y),)
