"""APIDataset: in-memory paired aptamer-protein data container."""

__author__ = ["nennomp", "siddharth7113"]
__all__ = ["APIDataset"]

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
    >>> list(ds.y)
    [1, 0]
    """

    _tags = {"scitype": "APIPairs"}

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
