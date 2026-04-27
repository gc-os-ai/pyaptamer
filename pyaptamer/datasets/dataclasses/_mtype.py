"""Input coercion for paired aptamer-protein data.

Internal module. ``coerce_input`` accepts any of the supported input shapes
and returns a canonical 2-column ``pd.DataFrame``. Used by
``APIDataset.from_any``.

Supported input shapes:
    - ``pd.DataFrame`` with columns ``["aptamer", "protein"]``
    - ``list[tuple[str, str]]`` of (aptamer, protein) pairs
    - ``tuple[np.ndarray, np.ndarray]`` of (apta, prot)
    - ``tuple[MoleculeLoader, MoleculeLoader]``
"""

__all__ = ["coerce_input"]

import numpy as np
import pandas as pd


def coerce_input(X) -> pd.DataFrame:
    """Coerce a supported input shape to the canonical 2-column DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X

    if isinstance(X, list) and all(isinstance(p, tuple) and len(p) == 2 for p in X):
        apta, prot = zip(*X, strict=False)
        return pd.DataFrame({"aptamer": list(apta), "protein": list(prot)})

    if isinstance(X, tuple) and len(X) == 2:
        a, b = X
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return pd.DataFrame({"aptamer": a, "protein": b})

        # MoleculeLoader pair (lazy import to avoid circular dep at module load)
        from pyaptamer.data import MoleculeLoader

        if isinstance(a, MoleculeLoader) and isinstance(b, MoleculeLoader):
            apta_seqs = a.to_df_seq()["sequence"].to_numpy()
            prot_seqs = b.to_df_seq()["sequence"].to_numpy()
            if len(apta_seqs) != len(prot_seqs):
                raise ValueError(
                    f"MoleculeLoader pair length mismatch: "
                    f"{len(apta_seqs)} aptamer rows vs {len(prot_seqs)} protein rows."
                )
            return pd.DataFrame({"aptamer": apta_seqs, "protein": prot_seqs})

    raise TypeError(
        f"Cannot coerce object of type {type(X).__name__} to pd.DataFrame; "
        f"expected pd.DataFrame, list[tuple], (np.ndarray, np.ndarray), "
        f"or (MoleculeLoader, MoleculeLoader)."
    )
