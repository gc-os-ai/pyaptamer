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
from pyaptamer.trafos.encode import KMerFrequencies


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


def _wide(X):
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
        "input_type",
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

    def test_wide_frame_keeps_other_columns(self, object_instance):
        """A univariate transformer on a wide frame returns the other columns unchanged."""  # noqa: E501
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        args = _scenario_for(object_instance).args
        X = _wide(args["transform"]["X"])
        Xt = object_instance.fit(_wide(args["fit"]["X"])).transform(X)
        assert Xt.columns[0] == "round"
        assert Xt.columns[-1] == "gc"
        pd.testing.assert_series_equal(Xt["round"], X["round"].loc[Xt.index])
        pd.testing.assert_series_equal(Xt["gc"], X["gc"].loc[Xt.index])

    def test_wide_frame_matches_single_column(self, object_instance):
        """The transformed part of a wide frame equals the single-column output, renamed."""  # noqa: E501
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        args = _scenario_for(object_instance).args
        X_fit, X = args["fit"]["X"], args["transform"]["X"]
        column = X.columns[0]
        narrow = object_instance.fit(X_fit).transform(X)
        wide = object_instance.fit(_wide(X_fit)).transform(_wide(X))
        middle = wide.drop(columns=["round", "gc"])
        if narrow.shape[1] == 1:
            expected = narrow.set_axis([column], axis=1)
        else:
            names = [f"{column}__{c}" for c in narrow.columns]
            expected = narrow.set_axis(names, axis=1)
        pd.testing.assert_frame_equal(middle, expected)

    def test_wide_frame_string_column_with_missing_head(self, object_instance):
        """A string column is still found when its first rows are missing."""
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        args = _scenario_for(object_instance).args
        X = _wide(args["fit"]["X"])
        column = args["fit"]["X"].columns[0]
        X = pd.concat([pd.DataFrame({"round": [4], column: [None], "gc": [0.5]}), X])
        assert object_instance._select_column(X) == column

    def test_wide_frame_raises_on_two_string_columns(self, object_instance):
        """Two string columns and no ApplyToCols is an error naming both columns."""
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        X = _wide(_scenario_for(object_instance).args["fit"]["X"])
        X["id"] = "read"
        with pytest.raises(ValueError, match="ApplyToCols"):
            object_instance.fit(X)

    def test_wide_frame_raises_on_no_string_column(self, object_instance):
        """A wide frame without any string column is an error."""
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        X = pd.DataFrame({"round": [4, 4], "gc": [0.5, 0.4]})
        with pytest.raises(ValueError, match="no column"):
            object_instance.fit(X)

    def test_transform_needs_the_fitted_column(self, object_instance):
        """transform on a frame without the column found in fit raises."""
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        args = _scenario_for(object_instance).args
        object_instance.fit(_wide(args["fit"]["X"]))
        X = _wide(args["transform"]["X"]).rename(columns={"seq": "reads"})
        with pytest.raises(ValueError, match="fitted on column"):
            object_instance.transform(X)

    def test_transform_wide_after_narrow_fit_raises(self, object_instance):
        """Fitting on one column and transforming a wide frame is a clear error."""
        if object_instance.get_tag("capability:multivariate", False):
            pytest.skip("multivariate transformers see the whole frame")
        args = _scenario_for(object_instance).args
        object_instance.fit(args["fit"]["X"])
        with pytest.raises(ValueError, match="one-column frame"):
            object_instance.transform(_wide(args["transform"]["X"]))


def test_stateful_transform_uses_fitted_state():
    """A transformer with fitted state can read that state back in transform.

    Uses the local ``_RowCounter``, which stores the number of rows seen in
    fit and repeats it for every row in transform.
    """
    Xt = _RowCounter().fit_transform(pd.DataFrame({"seq": ["ACGU", "GUAC"]}))
    assert Xt["n_rows"].tolist() == [2, 2]


def test_wide_frame_with_duplicate_index_and_no_dropped_rows():
    """A repeated row index is fine when the transformer keeps every row."""
    X = pd.DataFrame({"seq": ["ACGT", "GGCC"], "round": [1, 2]}, index=[0, 0])
    Xt = KMerFrequencies(k=1).fit_transform(X)
    assert Xt["round"].tolist() == [1, 2]


def test_wide_frame_output_name_clash_raises():
    """A column already named like an output column is an error, not a duplicate."""
    X = pd.DataFrame({"seq": ["ACGT"], "seq__0": [1.0]})
    with pytest.raises(ValueError, match="seq__0"):
        KMerFrequencies(k=1).fit_transform(X)
