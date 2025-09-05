from pyaptamer.benchmarking.preprocessors import BasePreprocessor
from pyaptamer.utils._aptanet_utils import rna2dna


class AptaNetPreprocessor(BasePreprocessor):
    def _transform(self, df):
        df = df.copy()
        df["aptamer"] = df["aptamer"].apply(rna2dna)
        df["X"] = list(zip(df["aptamer"], df["protein"], strict=False))
        return df[["X", "y"]]
