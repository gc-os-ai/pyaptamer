"""AptaNet features generation."""

import numpy as np
import pandas as pd

from pyaptamer.pseaac import AptaNetPSeAAC
from pyaptamer.trafos.base import BaseTransform
from pyaptamer.trafos.features._kmer import KMerFeatures


class AptaNetFeatures(BaseTransform):
    """
    Convert (aptamer_sequence, protein_sequence) pairs into feature vectors.

    This function generates feature vectors for each (aptamer, protein) pair using:
    - k-mer representation of the aptamer sequence
    - Pseudo amino acid composition (PSeAAC) representation of the protein sequence

    Parameters
    ----------
    k : int, optional
        The k-mer size used to generate the k-mer vector from the aptamer sequence.
        Default is 4.
    """

    _tags = {
        "authors": ["satvshr"],
        "maintainers": ["satvshr"],
        "output_type": "numeric",
        "property:fit_is_empty": True,
        "capability:multivariate": True,
    }

    def __init__(self, k=4):
        self.k = k
        super().__init__()

    def _transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform. Expected to have 'aptamer' and 'protein' columns.

        Returns
        -------
        X : pd.DataFrame, shape (n_samples, n_features_transformed)
            Transformed data.
        """
        pseaac = AptaNetPSeAAC()
        kmer_encoder = KMerFeatures(k=self.k)

        if "aptamer" in X.columns and "protein" in X.columns:
            aptamer_seqs = X["aptamer"]
            protein_seqs = X["protein"]
        elif X.shape[1] >= 2:
            aptamer_seqs = X.iloc[:, 0]
            protein_seqs = X.iloc[:, 1]
        else:
            raise ValueError(
                "X must have at least two columns: 'aptamer' and 'protein'."
            )

        # process kmers
        kmer_df = pd.DataFrame({"aptamer": aptamer_seqs})
        kmer_features = kmer_encoder.transform(kmer_df).values

        # process pseaac
        pseaac_features = []
        for protein_seq in protein_seqs:
            if not isinstance(protein_seq, str):
                protein_seq = ""
            pseaac_features.append(pseaac.transform(protein_seq))

        pseaac_features = np.vstack(pseaac_features)

        feats = np.concatenate([kmer_features, pseaac_features], axis=1)

        return pd.DataFrame(feats.astype(np.float32), index=X.index)

    def get_test_params(self):
        """Get test parameters for AptaNetFeatures."""
        param0 = {"k": 4}
        return [param0]
