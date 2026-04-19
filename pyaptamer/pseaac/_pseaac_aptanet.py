__author__ = ["nennomp", "satvshr"]
__all__ = ["AptaNetPSeAAC"]

from pyaptamer.trafos.pseaac import AptaNetPSeAAC as _AptaNetPSeAACTransformer

class AptaNetPSeAAC:
    """
    Compute Pseudo Amino Acid Composition (PseAAC) features for a protein sequence.
    
    This is a backward-compatible wrapper for the AptaNetPSeAAC transformer in trafos.
    """
    
    def __init__(self, **kwargs):
        self._transformer = _AptaNetPSeAACTransformer(**kwargs)

    def transform(self, protein_sequence):
        """Generate the PseAAC feature vector for the given protein sequence."""
        import pandas as pd
        # Handle string input as before
        if isinstance(protein_sequence, str):
            df = pd.DataFrame({"sequence": [protein_sequence]})
            Xt = self._transformer.transform(df)
            return Xt.values[0]
        return self._transformer.transform(protein_sequence)
