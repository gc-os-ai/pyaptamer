"""Test collection for all BaseTransform transformers in pyaptamer.

skbase's ``BaseFixtureGenerator`` finds the transformers and builds one test
instance per ``get_test_params`` entry; ``TestAllObjects`` adds the skbase
tests, and the fitted-state tests below run with them.
Scenarios give each transformer its fit/transform input data, matched by tags.
"""

__author__ = ["siddharth7113"]

import pandas as pd
import pytest
from skbase._exceptions import NotFittedError
from skbase.testing import BaseFixtureGenerator, TestAllObjects

from pyaptamer.data import MoleculeLoader
from pyaptamer.trafos.base import BaseTransform


class _RowCounter(BaseTransform):
    """Transformer with fitted state, covering the non-empty ``_fit`` branch.

    No transformer in the package currently has fitted state, so the branch
    of ``fit`` that calls ``_fit`` would otherwise go untested.

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


def _scenarios():
    """All scenarios the tests can use."""
    return [_SequenceFrameScenario(), _MoleculePairsScenario()]


def _scenario_for(obj):
    """The first scenario that fits obj's class; test_scenario_applies checks one exists."""  # noqa: E501
    return next(s for s in _scenarios() if s.is_applicable(type(obj)))


class PackageConfig:
    """Config that the skbase test classes read."""

    package_name = "pyaptamer"

    # all tags used in the package; test_object_tags fails on unlisted tags
    valid_tags = [
        "object_type",
        "authors",
        "maintainers",
        "capability:y",
        "capability:multivariate",
        "property:fit_is_empty",
        "property:elementwise",
        "output_type",
    ]


class TransformerFixtureGenerator(PackageConfig, BaseFixtureGenerator):
    """Creates the object_class and object_instance test arguments.

    Classes come from searching pyaptamer for BaseTransform subclasses, plus
    the local _RowCounter, which the search cannot find. Instances are built
    fresh for every test, one per get_test_params entry.
    """

    object_type_filter = BaseTransform

    def _all_objects(self):
        return super()._all_objects() + [_RowCounter]


class TestAllTransformers(TransformerFixtureGenerator, TestAllObjects):
    """The fitted-state tests, plus the standard skbase tests by inheritance."""

    def test_scenario_applies(self, object_class):
        """Every transformer is matched by at least one scenario, so none goes unchecked."""  # noqa: E501
        assert any(s.is_applicable(object_class) for s in _scenarios())

    def test_not_fitted_before_fit(self, object_instance):
        """A freshly constructed transformer reports itself as not fitted."""
        assert object_instance.is_fitted is False
        with pytest.raises(NotFittedError, match="has not been fitted"):
            object_instance.check_is_fitted()

    def test_raises_not_fitted_error(self, object_instance):
        """transform before fit raises NotFittedError"""
        scenario = _scenario_for(object_instance)
        with pytest.raises(NotFittedError, match="has not been fitted"):
            object_instance.transform(**scenario.args["transform"])

    def test_fit_sets_is_fitted(self, object_instance):
        """fit returns self and marks the transformer fitted, even if fit_is_empty."""
        scenario = _scenario_for(object_instance)
        assert object_instance.fit(**scenario.args["fit"]) is object_instance
        assert object_instance.is_fitted is True
        object_instance.check_is_fitted()

    def test_fit_transform_sets_is_fitted(self, object_instance):
        """fit_transform leaves the transformer in a fitted state."""
        scenario = _scenario_for(object_instance)
        object_instance.fit_transform(**scenario.args["fit"])
        assert object_instance.is_fitted is True


def test_stateful_transform_uses_fitted_state():
    """A transformer with fitted state can read that state back in transform."""
    Xt = _RowCounter().fit_transform(pd.DataFrame({"seq": ["ACGU", "GUAC"]}))
    assert Xt["n_rows"].tolist() == [2, 2]
