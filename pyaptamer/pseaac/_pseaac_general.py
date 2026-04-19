__author__ = ["nennomp", "satvshr"]
__all__ = ["PSeAAC"]

from pyaptamer.trafos.pseaac import PSeAAC as _PSeAACTransformer

class PSeAAC:
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.
    
    This is a backward-compatible wrapper for the PSeAAC transformer in trafos.
    """
    
    def __init__(self, **kwargs):
        self._transformer = _PSeAACTransformer(**kwargs)

    def transform(self, protein_sequence):
        """Generate the PseAAC feature vector for the given protein sequence."""
        import pandas as pd
        # Handle string input as before
        if isinstance(protein_sequence, str):
            df = pd.DataFrame({"sequence": [protein_sequence]})
            Xt = self._transformer.transform(df)
            return Xt.values[0]
        return self._transformer.transform(protein_sequence)
