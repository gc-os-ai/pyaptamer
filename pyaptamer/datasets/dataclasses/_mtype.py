"""Mtype dispatcher for paired aptamer-protein data containers.

Internal module. Defines the supported mtypes and a `convert_to` function
that converts between them. Used by `BaseAptamerDataset` for input
coercion; not part of the public API.

Supported mtypes (round-trippable):
    - "pd.DataFrame": columns ["aptamer", "protein"]
    - "list_tuples": list[tuple[str, str]]
    - "numpy_arrays": tuple[np.ndarray, np.ndarray] of (apta, prot)

Input-only mtypes (coerce to pd.DataFrame, no reverse):
    - "MoleculeLoader_pair": tuple[MoleculeLoader, MoleculeLoader]
"""

__all__ = ["SUPPORTED_MTYPES", "INPUT_ONLY_MTYPES", "convert_to", "coerce_input"]

import numpy as np
import pandas as pd

SUPPORTED_MTYPES = ("pd.DataFrame", "list_tuples", "numpy_arrays")
INPUT_ONLY_MTYPES = ("MoleculeLoader_pair",)


def _detect_mtype(X):
    if isinstance(X, pd.DataFrame):
        return "pd.DataFrame"
    if isinstance(X, list) and all(isinstance(p, tuple) and len(p) == 2 for p in X):
        return "list_tuples"
    if (
        isinstance(X, tuple)
        and len(X) == 2
        and all(isinstance(arr, np.ndarray) for arr in X)
    ):
        return "numpy_arrays"
    return None


def _df_to_list_tuples(df):
    return list(zip(df["aptamer"].tolist(), df["protein"].tolist(), strict=False))


def _df_to_numpy(df):
    return df["aptamer"].to_numpy(), df["protein"].to_numpy()


def _list_tuples_to_df(pairs):
    apta, prot = zip(*pairs, strict=False)
    return pd.DataFrame({"aptamer": list(apta), "protein": list(prot)})


def _numpy_to_df(arrays):
    apta, prot = arrays
    return pd.DataFrame({"aptamer": apta, "protein": prot})


_CONVERTERS = {
    ("pd.DataFrame", "list_tuples"): _df_to_list_tuples,
    ("pd.DataFrame", "numpy_arrays"): _df_to_numpy,
    ("list_tuples", "pd.DataFrame"): _list_tuples_to_df,
    ("numpy_arrays", "pd.DataFrame"): _numpy_to_df,
}


def convert_to(X, to_mtype):
    """Convert X to the requested mtype.

    Parameters
    ----------
    X : pd.DataFrame, list of (str, str) tuples, or (np.ndarray, np.ndarray) tuple
    to_mtype : str
        One of SUPPORTED_MTYPES.

    Returns
    -------
    The same data in the requested mtype. If X is already in to_mtype,
    returns X unchanged (identity, not copy).

    Raises
    ------
    ValueError if to_mtype is not in SUPPORTED_MTYPES.
    TypeError if X is not in any recognized mtype.
    """
    if to_mtype not in SUPPORTED_MTYPES:
        raise ValueError(
            f"Unknown to_mtype {to_mtype!r}; expected one of {SUPPORTED_MTYPES}"
        )

    src = _detect_mtype(X)
    if src is None:
        raise TypeError(
            f"Cannot convert object of type {type(X).__name__} — "
            f"expected one of {SUPPORTED_MTYPES}"
        )

    if src == to_mtype:
        return X

    if (src, to_mtype) in _CONVERTERS:
        return _CONVERTERS[(src, to_mtype)](X)

    df = _CONVERTERS[(src, "pd.DataFrame")](X)
    return _CONVERTERS[("pd.DataFrame", to_mtype)](df)


def _is_molecule_loader_pair(X):
    from pyaptamer.data import MoleculeLoader

    return (
        isinstance(X, tuple)
        and len(X) == 2
        and all(isinstance(m, MoleculeLoader) for m in X)
    )


def _molecule_loader_pair_to_df(pair):
    apta_loader, prot_loader = pair
    apta_df = apta_loader.to_df_seq()
    prot_df = prot_loader.to_df_seq()
    apta_seqs = apta_df["sequence"].to_numpy()
    prot_seqs = prot_df["sequence"].to_numpy()
    if len(apta_seqs) != len(prot_seqs):
        raise ValueError(
            f"MoleculeLoader pair length mismatch: "
            f"{len(apta_seqs)} aptamer rows vs {len(prot_seqs)} protein rows."
        )
    return pd.DataFrame({"aptamer": apta_seqs, "protein": prot_seqs})


def coerce_input(X):
    """Coerce an input to the canonical pd.DataFrame mtype.

    Accepts any SUPPORTED_MTYPE plus the input-only MoleculeLoader_pair.
    Returns the input unchanged if already a pd.DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        return X
    if _is_molecule_loader_pair(X):
        return _molecule_loader_pair_to_df(X)
    if _detect_mtype(X) is not None:
        return convert_to(X, to_mtype="pd.DataFrame")
    raise TypeError(
        f"Cannot coerce object of type {type(X).__name__} to pd.DataFrame; "
        f"expected one of {SUPPORTED_MTYPES + INPUT_ONLY_MTYPES}"
    )
