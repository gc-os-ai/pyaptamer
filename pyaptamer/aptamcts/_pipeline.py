"""AptaMCTS pipeline for aptamer-protein interaction prediction."""

__author__ = ["aditi-dsi", "siddharth7113"]
__all__ = ["AptaMCTSPipeline"]

import pandas as pd
from skbase.base import BaseEstimator
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from pyaptamer.data import MoleculeLoader
from pyaptamer.transformations.compose import ApplyToCols
from pyaptamer.transformations.encode import ConjointTriad, KMerFrequencies
from pyaptamer.transformations.transform import DNAToRNA


class AptaMCTSPipeline(BaseEstimator):
    """AptaMCTS classifier for aptamer-protein interaction prediction [1]_.

    AptaMCTS predicts whether an aptamer binds a protein from the two
    sequences alone. It is designed for RNA aptamers. A DNA sequence is
    converted to RNA with ``DNAToRNA``, which replaces T with U, before
    encoding. The aptamer is encoded with ``KMerFrequencies`` over ``ACGU``,
    k-mers of length 1 to ``rna_k``, 340 values by default. The protein is
    encoded with ``ConjointTriad``, k-mers of length 1 to ``prot_k`` over the
    7 conjoint triad groups, 399 values by default. Both blocks are divided by the
    sequence length, which the Apta-MCTS code calls iCTF [2]_. The features
    go to a random forest with balanced class weights, as in the original
    code.

    A fitted pipeline is the scoring function of
    ``pyaptamer.experiments.AptamerEvalAptaMCTS``, which drives
    ``pyaptamer.mcts.MCTS`` to generate candidate aptamers for a target.

    The pipeline takes a ``MoleculeLoader`` or a ``DataFrame`` with an
    aptamer column and a protein column. Other columns are ignored.

    Parameters
    ----------
    rna_k : int, default=4
        Longest k-mer counted in the aptamer sequence.
    prot_k : int, default=3
        Longest k-mer counted in the protein sequence.
    aptamer_col : str, default="aptamer"
        Name of the column holding aptamer sequences.
    protein_col : str, default="protein"
        Name of the column holding protein sequences.
    estimator : sklearn classifier or None, default=None
        Classifier applied to the features. If None, uses
        ``RandomForestClassifier(class_weight="balanced")``.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Steps ``to_rna``, ``aptamer``, ``protein`` and ``clf``.

    References
    ----------
    .. [1] Lee, G. et al. Predicting aptamer sequences that interact with
       target proteins using an aptamer-protein interaction classifier and a
       Monte Carlo tree search approach. *PLoS ONE* 16, e0253760 (2021).
       https://doi.org/10.1371/journal.pone.0253760
    .. [2] https://github.com/leekh7411/Apta-MCTS

    Examples
    --------
    The li2014 benchmark data of the paper is bundled as a ``MoleculeLoader``
    of aptamer and protein sequences with a label column:

    >>> from pyaptamer.aptamcts import AptaMCTSPipeline
    >>> from pyaptamer.datasets import load_li2014
    >>> X, y = load_li2014(split="test", return_X_y=True)
    >>> pipe = AptaMCTSPipeline().fit(X, y["label"])
    >>> labels = pipe.predict(X)
    >>> proba = pipe.predict_proba(X)

    Any ``MoleculeLoader`` works, so the aptamer and protein cells can be
    paths to sequence files. See the user guide for an example.
    """

    def __init__(
        self,
        rna_k=4,
        prot_k=3,
        aptamer_col="aptamer",
        protein_col="protein",
        estimator=None,
    ):
        self.rna_k = rna_k
        self.prot_k = prot_k
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        self.estimator = estimator
        super().__init__()

    def _build_pipeline(self):
        aptamer = KMerFrequencies(k=self.rna_k, alphabet="ACGU", normalize="length")
        protein = ConjointTriad(k=self.prot_k)
        if self.estimator is None:
            estimator = RandomForestClassifier(class_weight="balanced")
        else:
            estimator = clone(self.estimator)
        return Pipeline(
            [
                ("to_rna", ApplyToCols(DNAToRNA(), cols=self.aptamer_col)),
                ("aptamer", ApplyToCols(aptamer, cols=self.aptamer_col)),
                ("protein", ApplyToCols(protein, cols=self.protein_col)),
                ("clf", estimator),
            ]
        )

    def _to_frame(self, X):
        if isinstance(X, MoleculeLoader):
            X = X.to_dataframe()
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a MoleculeLoader instance or a pandas DataFrame. "
                f"Got {type(X)} instead."
            )
        columns = [self.aptamer_col, self.protein_col]
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"X has no column {missing}. X has columns {list(X.columns)}. "
                "Pass the names as AptaMCTSPipeline(aptamer_col=..., protein_col=...)."
            )
        return X[columns]

    def fit(self, X, y):
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(self._to_frame(X), y)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        self.check_is_fitted(method_name="predict_proba")
        return self.pipeline_.predict_proba(self._to_frame(X))

    def predict(self, X):
        self.check_is_fitted(method_name="predict")
        return self.pipeline_.predict(self._to_frame(X))
