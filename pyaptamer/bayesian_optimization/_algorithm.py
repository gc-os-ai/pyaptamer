"""Bayesian Optimization for suggesting new points in a latent space."""

__all__ = ["BayesianOptimizer"]

import numpy as np
from skbase.base import BaseObject
from skopt import Optimizer


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
    """

    def __init__(
        self,
        bounds,
        batch_size=10,
        minimize=True,
        kappa=2.0,
        n_restarts=10,
        random_state=None,
    ):
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be >= 1, got {batch_size}.")
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

    def suggest(self, X, y):
        """Suggest the next batch of points to evaluate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dims)
            Coordinates of the points that were already evaluated.
        y : array-like of shape (n_samples,)
            Scores of the evaluated points.

        Returns
        -------
        np.ndarray of shape (batch_size, n_dims)
            The suggested points, inside `bounds`.
        """
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

        opt = Optimizer(
            dimensions=[(float(low), float(high)) for low, high in bounds],
            base_estimator="GP",  # Gaussian process, like GPyOpt
            acq_func="LCB",
            acq_func_kwargs={"kappa": self.kappa},
            random_state=self.random_state,
            n_initial_points=1,
        )
        opt.tell(X.tolist(), y_min.tolist())
        points = opt.ask(n_points=self.batch_size, strategy="cl_min")

        self.next_points_ = np.asarray(points)
        return self.next_points_
