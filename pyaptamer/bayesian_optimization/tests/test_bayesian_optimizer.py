"""Tests for BayesianOptimizer."""

import numpy as np
import pytest

from pyaptamer.bayesian_optimization import BayesianOptimizer


def _obj_fc(X):
    """Defined objective function: minimum 0 at the origin."""
    return (X**2).sum(axis=1)


@pytest.fixture
def evaluated():
    """Eight already-evaluated points in three dimensions."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(8, 3))
    return X, _obj_fc(X)


# --- valid inputs tests-------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"kappa": -1.0},
    ],
)
def test_invalid_parameters(kwargs):
    """Test to make sure Invalid settings raise a ValueError"""
    with pytest.raises(ValueError):
        BayesianOptimizer(bounds=(-1, 1), **kwargs)


def test_invalid_inputs(evaluated):
    """Test to make sure invalid inputs raise a ValueError"""
    X, y = evaluated
    bo = BayesianOptimizer((-2, 2), batch_size=2)
    with pytest.raises(ValueError, match="same number of rows"):
        bo.suggest(X, y[:-1])
    with pytest.raises(ValueError, match="2D"):
        bo.suggest(X[:, 0], y)
    with pytest.raises(ValueError, match="at least one"):
        bo.suggest(np.empty((0, 3)), np.empty(0))


@pytest.mark.parametrize(
    "bounds",
    [[(-1, 1), (-1, 1)], (1, -1), (0, 0)],
)
def test_invalid_bounds(evaluated, bounds):
    """Bounds that do not fit the data or are reversed raise a ValueError."""
    X, y = evaluated
    bo = BayesianOptimizer(bounds, batch_size=2)
    with pytest.raises(ValueError):
        bo.suggest(X, y)


@pytest.mark.parametrize("bounds", [(-1, 1), [(-1, 1), (0, 2), (-3, -2)]])
def test_suggestions_inside_bounds(bounds):
    """Every suggested coordinate lies inside the requested range."""
    expected = np.asarray(bounds, dtype=float)
    if expected.shape == (2,):
        expected = np.tile(expected, (3, 1))

    rng = np.random.default_rng(0)
    X = rng.uniform(expected[:, 0], expected[:, 1], size=(8, 3))
    bo = BayesianOptimizer(bounds=bounds, batch_size=5, random_state=0)
    points = bo.suggest(X, _obj_fc(X))

    assert np.all(points >= expected[:, 0])
    assert np.all(points <= expected[:, 1])


def test_points_outside_bounds(evaluated):
    """Evaluated points outside the search range raise a ValueError."""
    X, y = evaluated
    bo = BayesianOptimizer((-1, 1), batch_size=2)  # X must go outside of the range
    with pytest.raises(ValueError, match="outside"):
        bo.suggest(X, y)


# --- valid usage tests -----


def test_suggestions_are_unique(evaluated):
    """A batch does not contain the same point twice."""
    X, y = evaluated
    bo = BayesianOptimizer((-2, 2), batch_size=5, random_state=0)
    points = bo.suggest(X, y)
    assert len(np.unique(points, axis=0)) == 5


def test_stores_next_points(evaluated):
    """The last suggestion is stored in `next_points_`."""
    X, y = evaluated
    bo = BayesianOptimizer((-2, 2), batch_size=2, random_state=0)
    points = bo.suggest(X, y)
    np.testing.assert_array_equal(bo.next_points_, points)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("n_dims", [1, 2, 5])
def test_suggest_shape(batch_size, n_dims):
    """Output has one row per suggestion and one column per dimension."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(6, n_dims))
    bo = BayesianOptimizer(bounds=(-2, 2), batch_size=batch_size, random_state=0)
    points = bo.suggest(X, _obj_fc(X))
    assert points.shape == (batch_size, n_dims)
