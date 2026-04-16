"""Base class for aptamer-protein paired data containers."""

__all__ = ["BaseAptamerDataset"]

import numpy as np
import pandas as pd
from skbase.base import BaseObject

from pyaptamer.datasets.dataclasses._mtype import coerce_input  # noqa: F401


class BaseAptamerDataset(BaseObject):
    """Base class for paired aptamer-protein in-memory data containers.

    Holds raw sequence strings only (no encoding). Concrete subclasses
    set their own ``scitype`` tag.

    Parameters
    ----------
    x_apta, x_prot : array-like or convertible to pd.DataFrame
        Aptamer and protein sequences. Accepted shapes match the
        SUPPORTED_MTYPES + INPUT_ONLY_MTYPES of ``_mtype``.
    y : array-like, optional
        Labels aligned with the rows of x_apta/x_prot. ``None`` if unlabeled.

    Attributes
    ----------
    y : np.ndarray or None
        Labels, stored as a numpy array (sklearn convention).
    """

    _tags = {
        "object_type": "dataset",
        "authors": ["siddharth7113"],
        "maintainers": [],
        "python_dependencies": None,
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
        """Return X in the canonical inner mtype (pd.DataFrame)."""
        return self._X

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
