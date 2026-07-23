"""Contract tests for BaseTransform descendants.

Checks the fitted-state and sklearn tag contract against every transformer in
the package:

    test_scenario_applies             - every transformer is matched by a scenario
    test_not_fitted_before_fit        - is_fitted is False before fit
    test_raises_not_fitted_error      - transform before fit raises NotFittedError
    test_fit_sets_is_fitted           - fit sets is_fitted and returns self
    test_fit_transform_sets_is_fitted - fit_transform leaves the state fitted
    test_sklearn_tags                 - sklearn reads tags off the transformer

Transformers are discovered with ``all_objects``, and the input each is checked
against comes from a scenario, after sktime's test design: a scenario supplies
the ``args`` for ``fit`` and ``transform``, and declares in ``is_applicable``
which transformers it applies to, by reading their tags.

Adding a transformer therefore needs no change here, as long as a scenario
applies to it - ``test_scenario_applies`` fails if none does. A transformer
taking a new kind of input needs a new scenario class.
"""

__author__ = ["siddharth7113"]

import warnings

import pandas as pd
import pytest
from skbase._exceptions import NotFittedError
from skbase.lookup import all_objects
from sklearn.utils import Tags, get_tags

from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.base import BaseTransform


class _RowCounter(BaseTransform):
    """Transformer with fitted state, covering the non-empty ``_fit`` branch.

    No transformer in the package currently has fitted state, so the branch of
    ``fit`` dispatching to ``_fit`` would otherwise go untested.

    Note: Remove this when a transformer covering this case is present.
    """

    _tags = {"property:fit_is_empty": False}

    def _fit(self, X, y=None):
        self.n_rows_ = len(X)
        return self

    def _transform(self, X):
        return pd.DataFrame({"n_rows": [self.n_rows_] * len(X)}, index=X.index)


class _SequenceFrameScenario:
    """A single column of sequence strings, for univariate transformers."""

    def is_applicable(self, cls):
        return not cls.get_class_tag("capability:multivariate", False)

    @property
    def args(self):
        return {
            "fit": {"X": pd.DataFrame({"seq": ["ACGU", "GUAC"]})},
            "transform": {"X": pd.DataFrame({"seq": ["GUAC", "ACGU"]})},
        }


class _MoleculePairsScenario:
    """A MoleculeLoader of (aptamer, protein) pairs, for multivariate transformers."""

    def is_applicable(self, cls):
        return cls.get_class_tag("capability:multivariate", False)

    @property
    def args(self):
        def loader():
            return MoleculeLoader(
                data={
                    "aptamer": ["AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"],
                    "protein": ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"],
                }
            )

        return {"fit": {"X": loader()}, "transform": {"X": loader()}}


def _transformers():
    """Every transformer in the package, plus the local stateful one."""
    found = all_objects(
        package_name="pyaptamer",
        object_types=BaseTransform,
        return_names=False,
    )
    return found + [_RowCounter]


def _scenarios():
    """Every transformer scenario known to the contract tests."""
    return [_SequenceFrameScenario(), _MoleculePairsScenario()]


def _cases():
    """(transformer, scenario) pairs the contract is checked on."""
    return [
        pytest.param(cls, scenario, id=f"{cls.__name__}-{type(scenario).__name__}")
        for cls in _transformers()
        for scenario in _scenarios()
        if scenario.is_applicable(cls)
    ]


@pytest.mark.parametrize("cls", _transformers(), ids=lambda cls: cls.__name__)
def test_scenario_applies(cls):
    """Every transformer is matched by at least one scenario, so none goes unchecked."""
    assert any(scenario.is_applicable(cls) for scenario in _scenarios())


@pytest.mark.parametrize("cls", _transformers(), ids=lambda cls: cls.__name__)
def test_not_fitted_before_fit(cls):
    """A freshly constructed transformer reports itself as not fitted."""
    est = cls.create_test_instance()
    assert est.is_fitted is False
    with pytest.raises(NotFittedError, match="has not been fitted"):
        est.check_is_fitted()


@pytest.mark.parametrize("cls", _transformers(), ids=lambda cls: cls.__name__)
def test_sklearn_tags(cls):
    """sklearn reads tags off a transformer instead of falling back to defaults."""
    est = cls.create_test_instance()
    with warnings.catch_warnings(action="error", category=DeprecationWarning):
        tags = get_tags(est)
    assert isinstance(tags, Tags)
    assert tags.estimator_type == "transformer"
    assert tags.transformer_tags is not None


@pytest.mark.parametrize("cls, scenario", _cases())
def test_raises_not_fitted_error(cls, scenario):
    """transform before fit raises NotFittedError, not a bare AttributeError."""
    est = cls.create_test_instance()
    with pytest.raises(NotFittedError, match="has not been fitted"):
        est.transform(**scenario.args["transform"])


@pytest.mark.parametrize("cls, scenario", _cases())
def test_fit_sets_is_fitted(cls, scenario):
    """fit returns self and marks the transformer fitted, even if fit_is_empty."""
    est = cls.create_test_instance()
    assert est.fit(**scenario.args["fit"]) is est
    assert est.is_fitted is True
    est.check_is_fitted()


@pytest.mark.parametrize("cls, scenario", _cases())
def test_fit_transform_sets_is_fitted(cls, scenario):
    """fit_transform leaves the transformer in a fitted state."""
    est = cls.create_test_instance()
    est.fit_transform(**scenario.args["fit"])
    assert est.is_fitted is True


def test_stateful_transform_uses_fitted_state():
    """A transformer with fitted state can read that state back in transform."""
    Xt = _RowCounter().fit_transform(pd.DataFrame({"seq": ["ACGU", "GUAC"]}))
    assert Xt["n_rows"].tolist() == [2, 2]
