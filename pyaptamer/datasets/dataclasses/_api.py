"""APIDataset: in-memory paired aptamer-protein data container."""

__author__ = ["nennomp", "siddharth7113"]
__all__ = ["APIDataset"]

import numpy as np
import pandas as pd

from pyaptamer.datasets.dataclasses._base import BaseAptamerDataset
from pyaptamer.datasets.dataclasses._mtype import coerce_input


class APIDataset(BaseAptamerDataset):
    """In-memory container for aptamer-protein interaction (API) pairs.

    Holds raw sequence strings; performs no encoding. Encoding is the
    responsibility of downstream pipelines or transforms.

    Parameters
    ----------
    x_apta, x_prot : array-like
        Aptamer and protein sequences. May be np.ndarray, list, pd.Series,
        or single-column pd.DataFrame. Lengths must match.
    y : array-like, optional
        Labels aligned with the rows. Pass ``None`` for unlabeled data.

    Examples
    --------
    >>> from pyaptamer.datasets.dataclasses import APIDataset
    >>> ds = APIDataset(x_apta=["ACGU", "UGCA"], x_prot=["MKV", "LKR"], y=[1, 0])
    >>> ds.load().shape
    (2, 2)
    >>> ds.y.tolist()
    [1, 0]
    """

    _tags = {
        "scitype": "APIPairs",
        "X_inner_mtype": ["pd.DataFrame", "list_tuples", "numpy_arrays"],
        "has_y": True,
    }

    def __init__(self, x_apta=None, x_prot=None, y=None):
        super().__init__()
        if x_apta is None and x_prot is None:
            self._X = None
            self.y = None
            return
        self._X = self._check_inputs(x_apta, x_prot)
        self.y = np.asarray(y) if y is not None else None

    def load(self) -> pd.DataFrame:
        """Return X as a 2-column pd.DataFrame with columns [aptamer, protein]."""
        return self._X

    @classmethod
    def from_any(cls, X, y=None) -> "APIDataset":
        """Construct an APIDataset from any supported input shape.

        Accepted X shapes:
            - APIDataset (returned unchanged; pass-through)
            - pd.DataFrame with columns ["aptamer", "protein"]
            - list[tuple[str, str]] of (aptamer, protein) pairs
            - tuple[np.ndarray, np.ndarray] of (x_apta, x_prot)
            - tuple[MoleculeLoader, MoleculeLoader] (auto-coerces via .to_df_seq())

        Parameters
        ----------
        X : object
            Input in any of the supported shapes.
        y : array-like, optional
            Labels. Ignored if X is already an APIDataset.

        Returns
        -------
        APIDataset
        """
        if isinstance(X, APIDataset):
            return X
        df = coerce_input(X)
        return cls(x_apta=df["aptamer"], x_prot=df["protein"], y=y)

    def _check_inputs(self, x_apta, x_prot) -> pd.DataFrame:
        """Coerce x_apta and x_prot into the canonical two-column DataFrame.

        Each of x_apta, x_prot may be a single-column object (np.ndarray,
        list, pd.Series, single-column pd.DataFrame). The result has columns
        ["aptamer", "protein"] and aligned rows.
        """
        apta_series = self._to_series(x_apta, name="aptamer")
        prot_series = self._to_series(x_prot, name="protein")
        if len(apta_series) != len(prot_series):
            raise ValueError(
                f"x_apta and x_prot must have equal length; "
                f"got {len(apta_series)} and {len(prot_series)}."
            )
        return pd.DataFrame({"aptamer": apta_series, "protein": prot_series})

    @staticmethod
    def _to_series(x, name):
        if isinstance(x, pd.Series):
            return x.reset_index(drop=True).rename(name)
        if isinstance(x, pd.DataFrame):
            if x.shape[1] != 1:
                raise ValueError(
                    f"Expected single-column DataFrame for {name}; "
                    f"got {x.shape[1]} columns."
                )
            return x.iloc[:, 0].reset_index(drop=True).rename(name)
        if isinstance(x, list | np.ndarray):
            return pd.Series(list(x), name=name)
        raise TypeError(
            f"Unsupported type for {name}: {type(x).__name__}; "
            f"expected np.ndarray, list, pd.Series, or single-column pd.DataFrame."
        )
