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
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from pyaptamer.data import MoleculeLoader
from pyaptamer.transformations.base import BaseTransform
from pyaptamer.transformations.compose import ApplyToCols


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
        first = "AAACGTACGTTGCAAGCTTGCAGTACGATCGATCGTATTT"
        second = "AAAGACCGGTAACGTTACGGATCCATGCATGCAAGTCTTT"
        return {
            "fit": {"X": pd.DataFrame({"seq": [first, second]})},
            "transform": {"X": pd.DataFrame({"seq": [second, first]})},
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


def _with_extra_columns(X):
    """The scenario frame with an int and a float column before and after it."""
    n = len(X)
    return pd.DataFrame(
        {"round": [4] * n, X.columns[0]: X.iloc[:, 0].tolist(), "gc": [0.5] * n},
        index=X.index,
    )


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
    """Tests every BaseTransform subclass must pass.

    The skbase tests come in through ``TestAllObjects``. The tests below
    check what ``BaseTransform`` promises for fit and transform: the fitted
    flag, input coercion, and the output frame.
    """

    def test_scenario_applies(self, object_class):
        """Every transformer class is matched by at least one scenario.

        A class without a scenario would have no input data and the tests
        below would fail with a StopIteration instead of a clear message.
        """
        assert any(s.is_applicable(object_class) for s in _scenarios())

    def test_not_fitted_before_fit(self, object_instance):
        """A freshly constructed transformer reports itself as not fitted.

        ``is_fitted`` is False and ``check_is_fitted`` raises NotFittedError.
        """
        assert object_instance.is_fitted is False
        with pytest.raises(NotFittedError, match="has not been fitted"):
            object_instance.check_is_fitted()

    def test_raises_not_fitted_error(self, object_instance):
        """transform before fit raises NotFittedError.

        This holds even for transformers with ``property:fit_is_empty``,
        whose fit does nothing but set the fitted flag.
        """
        scenario = _scenario_for(object_instance)
        with pytest.raises(NotFittedError, match="has not been fitted"):
            object_instance.transform(**scenario.args["transform"])

    def test_fit_sets_is_fitted(self, object_instance):
        """fit returns self and marks the transformer fitted.

        Holds for transformers with ``property:fit_is_empty`` too.
        """
        scenario = _scenario_for(object_instance)
        assert object_instance.fit(**scenario.args["fit"]) is object_instance
        assert object_instance.is_fitted is True
        object_instance.check_is_fitted()

    def test_output_is_frame_over_input_index(self, object_instance):
        """transform returns a DataFrame indexed by rows of the input.

        The output index is a subset of the input index. Transformers may
        drop rows (PrimerTrimmer with ``on_unmatched="drop"``); none of the
        current transformers adds rows.
        """
        args = _scenario_for(object_instance).args
        X = args["transform"]["X"]
        Xt = object_instance.fit(**args["fit"]).transform(X)
        assert isinstance(Xt, pd.DataFrame)
        X_index = X.to_dataframe().index if isinstance(X, MoleculeLoader) else X.index
        assert Xt.index.isin(X_index).all()

    def test_moleculeloader_matches_dataframe(self, object_instance):
        """A MoleculeLoader over a frame transforms the same as the frame.

        ``BaseTransform._check_X_y`` coerces a MoleculeLoader with
        ``to_dataframe()``. The scenario frame is wrapped in a loader and
        both inputs must give identical output.

        Skipped for multivariate transformers, whose scenario already feeds
        a MoleculeLoader.
        """
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("the multivariate scenario already uses a MoleculeLoader")
        args = _scenario_for(object_instance).args
        X = args["transform"]["X"]
        object_instance.fit(**args["fit"])
        from_frame = object_instance.transform(X)
        from_loader = object_instance.transform(MoleculeLoader(data=X))
        pd.testing.assert_frame_equal(from_frame, from_loader)

    def test_univariate_accepts_any_column_name(self, object_instance):
        """A univariate transformer works on its one column whatever it is called.

        The transformer must read the column by position, not by a fixed
        name, so the scenario frame is renamed to ``reads`` before fit and
        transform.
        """
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("only univariate transformers receive a single column")
        args = _scenario_for(object_instance).args
        X_fit = args["fit"]["X"].set_axis(["reads"], axis=1)
        X_transform = args["transform"]["X"].set_axis(["reads"], axis=1)
        object_instance.fit(X_fit).transform(X_transform)

    def test_several_columns_raise_pointing_to_apply_to_cols(self, object_instance):
        """A univariate transformer given several columns raises and names ApplyToCols."""  # noqa: E501
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        args = _scenario_for(object_instance).args
        with pytest.raises(ValueError, match="ApplyToCols"):
            object_instance.fit(_with_extra_columns(args["fit"]["X"]))
        object_instance.fit(args["fit"]["X"])
        with pytest.raises(ValueError, match="ApplyToCols"):
            object_instance.transform(_with_extra_columns(args["transform"]["X"]))

    def test_apply_to_cols_matches_single_column(self, object_instance):
        """ApplyToCols gives the one-column output, renamed, with the other columns unchanged."""  # noqa: E501
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        args = _scenario_for(object_instance).args
        X_fit, X = args["fit"]["X"], args["transform"]["X"]
        column = X.columns[0]
        narrow = object_instance.clone().fit(X_fit).transform(X)
        wrapper = ApplyToCols(object_instance, cols=column)
        Xt = wrapper.fit(_with_extra_columns(X_fit)).transform(_with_extra_columns(X))
        assert Xt.columns[0] == "round"
        assert Xt.columns[-1] == "gc"
        pd.testing.assert_series_equal(
            Xt["round"], _with_extra_columns(X)["round"].loc[Xt.index]
        )
        pd.testing.assert_series_equal(
            Xt["gc"], _with_extra_columns(X)["gc"].loc[Xt.index]
        )
        middle = Xt.drop(columns=["round", "gc"])
        if narrow.shape[1] == 1:
            expected = narrow.set_axis([column], axis=1)
        else:
            expected = narrow.set_axis(
                [f"{column}__{c}" for c in narrow.columns], axis=1
            )
        pd.testing.assert_frame_equal(middle, expected)

    def test_sklearn_sees_fitted_state(self, object_instance):
        """scikit-learn's check_is_fitted raises before fit and passes after."""
        with pytest.raises(SklearnNotFittedError):
            check_is_fitted(object_instance)
        object_instance.fit(**_scenario_for(object_instance).args["fit"])
        check_is_fitted(object_instance)

    def test_sklearn_pipeline_ending_in_transformer(self, object_instance):
        """A scikit-learn Pipeline ending in the transformer gives its output."""
        args = _scenario_for(object_instance).args
        X_fit, X = args["fit"]["X"], args["transform"]["X"]
        expected = object_instance.clone().fit(X_fit).transform(X)
        pipe = Pipeline([("step", object_instance)]).fit(X_fit)
        pd.testing.assert_frame_equal(pipe.transform(X), expected)


def test_stateful_transform_uses_fitted_state():
    """A transformer with fitted state can read that state back in transform.

    Uses the local ``_RowCounter``, which stores the number of rows seen in
    fit and repeats it for every row in transform.
    """
    Xt = _RowCounter().fit_transform(pd.DataFrame({"seq": ["ACGU", "GUAC"]}))
    assert Xt["n_rows"].tolist() == [2, 2]
