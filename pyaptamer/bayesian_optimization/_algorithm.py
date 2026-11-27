"""Bayesian Optimization for suggesting new points in a latent space."""

__author__ = ["NoorMajdoub"]
__all__ = ["BayesianOptimizer"]

import numbers

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from skbase.base import BaseObject
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class BayesianOptimizer(BaseObject):
    """
    Model-agnostic Bayesian optimization for suggesting the best next batch
    of points to evaluate in a latent space
    A gaussian process is fitted on the evaluated points.
    We choose the points with the lower confidence bound
    (LCB) acquisition function,
    which balances exploration and exploitation.

    Parameters
    ----------
    bounds : tuple or list of tuples
        Search range of the coordinates.
        e.g. ``(-4, 4)`` or ``[(-4, 4), (-2, 2)]``
    batch_size : int, optional, default=10
        Number of points suggested per call. Must be >= 1.
    minimize : bool, optional, default=True
        Whether lower scores are better.
        If False, higher scores are better.
    kappa : float, optional, default=2.0
        Exploration weight of the LCB.
        Larger values favour unexplored regions.
        Must be >= 0.
    random_state : int, optional, default=None
        Seed for reproducibility.


    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.bayesian_optimization import BayesianOptimizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-2, 2, size=(8, 2))  # points that were already evaluated
    >>> y = (X**2).sum(axis=1)  # their scores (lower is better)
    >>> bo = BayesianOptimizer(bounds=(-2, 2), batch_size=3, random_state=0)
    >>> bo.suggest(X, y).shape
    (3, 2)
    """

    def __init__(
        self,
        bounds,
        batch_size=10,
        minimize=True,
        kappa=2.0,
        n_restarts=5,
        random_state=None,
    ):
        if not isinstance(batch_size, numbers.Integral) or isinstance(batch_size, bool):
            raise TypeError(f"`batch_size` must be an int, got {type(batch_size)}.")
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be >= 1, got {batch_size}.")
        if not isinstance(kappa, numbers.Real) or isinstance(kappa, bool):
            raise TypeError(f"`kappa` must be a number, got {type(kappa)}.")
        if kappa < 0:
            raise ValueError(f"`kappa` must be >= 0, got {kappa}.")
        self.bounds = bounds
        self.batch_size = batch_size
        self.minimize = minimize
        self.kappa = kappa
        self.n_restarts = n_restarts
        self.random_state = random_state
        super().__init__()

    def _get_bounds(self, n_dims):
        """Return the bounds as an array of shape (n_dims, 2)."""
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape == (2,):
            bounds = np.tile(bounds, (n_dims, 1))
        elif bounds.shape != (n_dims, 2):
            raise ValueError(
                f"`bounds` must be (low, high) or {n_dims} pairs, got {bounds.shape}."
            )
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each bound must satisfy low < high.")
        return bounds

    def _fit_gp(self, X, y, seed):
        """Fit the Gaussian process on (X, y)."""
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=seed,
        )
        return gp.fit(X, y)

    def _estimate_lipschitz(self, gp, X, bounds, rng, n_lipschitz=500):
        """Estimate the Lipschitz constant of the objective."""
        n_dims = X.shape[1]
        points = np.vstack(
            [rng.uniform(bounds[:, 0], bounds[:, 1], (n_lipschitz, n_dims)), X]
        )
        grad = np.empty_like(points)
        for j in range(n_dims):
            step = np.zeros(n_dims)
            step[j] = 1e-4 * (bounds[j, 1] - bounds[j, 0])
            grad[:, j] = (gp.predict(points + step) - gp.predict(points - step)) / (
                2 * step[j]
            )
        lipschitz = np.sqrt((grad**2).sum(axis=1)).max()
        return lipschitz if lipschitz > 1e-7 else 10.0

    def _objective(self, gp, x, penalties):
        """Penalized acquisition, in log space, to be minimized.
        The acquisition is the LCB (written as ``-mean + kappa * std``, to be
        maximized), mapped with a softplus and a log, and multiplied by one
        penalizer for each point already in the batch.
        """
        mean, std = gp.predict(np.atleast_2d(x), return_std=True)
        acq = -mean + self.kappa * std
        softplus = np.where(acq > 30, acq, np.logaddexp(0.0, acq))
        value = -np.where(acq < -30, acq, np.log(np.maximum(softplus, 1e-300)))
        for center, radius, scale in penalties:
            dist = np.sqrt(((np.atleast_2d(x) - center) ** 2).sum(axis=-1))
            value = value - norm.logcdf((dist - radius) / scale)
        return value

    def _next_point(self, gp, penalties, bounds, rng, n_candidates=1000, n_anchors=5):
        """Minimize the penalized acquisition."""
        n_dims = bounds.shape[0]
        candidates = rng.uniform(bounds[:, 0], bounds[:, 1], (n_candidates, n_dims))
        values = self._objective(gp, candidates, penalties)
        order = np.argsort(values)
        best_x, best_value = candidates[order[0]], values[order[0]]

        def value_and_grad(x):
            # forward finite differences, computed in a single batch of predictions
            step = 1e-6
            points = np.vstack([x, x + step * np.eye(n_dims)])
            values = self._objective(gp, points, penalties)
            return float(values[0]), (values[1:] - values[0]) / step

        for x0 in candidates[order[:n_anchors]]:
            res = minimize(
                value_and_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds
            )
            if res.fun < best_value:
                best_x, best_value = res.x, res.fun
        return best_x

    def suggest(self, X, y):
        """Suggest the next batch of points to evaluate."""
        X = np.asarray(X, dtype=float)  # data points in latent space
        y = np.asarray(y, dtype=float).ravel()  # binding score
        if X.ndim != 2:
            raise ValueError(f"`X` must be 2D, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("`X` must contain at least one evaluated point.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("`X` and `y` must have the same number of rows.")
        bounds = self._get_bounds(X.shape[1])
        if np.any(X < bounds[:, 0]) or np.any(X > bounds[:, 1]):
            raise ValueError("`X` contains points outside `bounds`.")

        y_min = y if self.minimize else -y

        rng = np.random.default_rng(self.random_state)
        gp = self._fit_gp(X, y_min, seed=int(rng.integers(2**31 - 1)))

        batch = [self._next_point(gp, [], bounds, rng)]
        if self.batch_size > 1:
            lipschitz = self._estimate_lipschitz(gp, X, bounds, rng)
            best = y_min.min()
        while len(batch) < self.batch_size:
            # penalizers centred at the points chosen so far
            penalties = []
            for x_j in batch:
                mean, std = gp.predict(x_j[None, :], return_std=True)
                penalties.append(
                    (
                        x_j,
                        float((mean[0] - best) / lipschitz),
                        float(max(std[0], 1e-8) / lipschitz),
                    )
                )
            batch.append(self._next_point(gp, penalties, bounds, rng))

        self.next_points_ = np.vstack(batch)
        return self.next_points_
