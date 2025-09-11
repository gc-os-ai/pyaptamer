__author__ = "satvshr"
__all__ = ["AptaNetPreprocessor"]

from pyaptamer.benchmarking.preprocessors import BasePreprocessor
from pyaptamer.utils._aptanet_utils import rna2dna


class AptaNetPreprocessor(BasePreprocessor):
    """
    Preprocessor for AptaNet benchmarking.

    Converts a pandas DataFrame with columns "aptamer", "protein" and "y"
    into a DataFrame with columns "X" and "y" suitable for AptaNet pipelines.

    The returned "X" column contains tuples (aptamer_dna, protein) where
    aptamer sequences are converted from RNA -> DNA via :func:`rna2dna`.
    """

    def _transform(self, df):
        """
        Transform input DataFrame into (X, y) format.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with columns:
            - 'aptamer' : str or sequence-like (RNA allowed)
            - 'protein' : str (amino-acid sequence)
            - 'y' : target labels

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
            - X : list of tuples (aptamer_dna, protein)
            - y : original target column
        """
        df = df.copy()
        df["aptamer"] = df["aptamer"].apply(rna2dna)
        df["X"] = list(zip(df["aptamer"], df["protein"], strict=False))
        return df[["X", "y"]]
