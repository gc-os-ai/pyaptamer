__author__ = ["nennomp", "satvshr", "siddharth7113"]
__all__ = ["AptaNetPipeline"]
__required__ = ["python>=3.10"]

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from pyaptamer.aptanet import AptaNetClassifier
from pyaptamer.base import BasePipelineDelegator
from pyaptamer.transformations.encode import KMerFrequencies, PSeAAC


class AptaNetPipeline(BasePipelineDelegator):
    """
    AptaNet algorithm for aptamer–protein interaction prediction [1]_

    Implements the AptaNet algorithm, a deep learning method that combines
    sequence-derived features with RandomForest-based feature selection and a
    multi-layer perceptron to predict whether an aptamer and a protein interact
    (binary classification).

    The pipeline takes a MoleculeLoader or a DataFrame of aptamer/protein
    pairs. The aptamer column is encoded with `KMerFrequencies` and the
    protein column with `PSeAAC`, using the 21 physicochemical properties in
    7 groups of 3 as in AptaNet. The two feature blocks are concatenated and
    passed to the estimator.

    Parameters
    ----------
    k : int, optional, default=4
        The k-mer size used to generate aptamer k-mer vectors.
    aptamer_col : str, optional, default="aptamer"
        Name of the column holding aptamer sequences.
    protein_col : str, optional, default="protein"
        Name of the column holding protein sequences.
    estimator : sklearn-compatible estimator or None, default=None
        Estimator applied to the features, which are passed as float64. A
        raw torch network needs a wrapper that casts, such as
        `AptaNetClassifier`. If None, uses `AptaNetClassifier`.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Steps ``features`` (a ``ColumnTransformer``) and ``clf``.

    References
    ----------

    .. [1] Emami, N., Ferdousi, R. AptaNet as a deep learning approach for
        aptamer–protein interaction prediction. *Scientific Reports*, 11, 6074 (2021).
        https://doi.org/10.1038/s41598-021-85629-0
    .. [2] GitHub repository: https://github.com/nedaemami/AptaNet
    .. [3] PDF version of the article: https://www.nature.com/articles/s41598-021-85629-0.pdf

    Examples
    --------
    >>> from pyaptamer.aptanet import AptaNetPipeline
    >>> from pyaptamer.data import MoleculeLoader
    >>> import numpy as np
    >>> pipe = AptaNetPipeline()
    >>> aptamer_seq = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein_seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> X_train = MoleculeLoader(
    ...     data={"aptamer": [aptamer_seq] * 40, "protein": [protein_seq] * 40}
    ... )
    >>> y_train = np.array([0] * 20 + [1] * 20, dtype=np.float32)
    >>> X_test = MoleculeLoader(
    ...     data={"aptamer": [aptamer_seq] * 10, "protein": [protein_seq] * 10}
    ... )
    >>> pipe.fit(X_train, y_train)  # doctest: +ELLIPSIS
    AptaNetPipeline(...)
    >>> preds = pipe.predict(X_test)
    >>> proba = pipe.predict_proba(X_test)
    """

    _tags = {
        "authors": ["nennomp", "satvshr", "siddharth7113"],
        "maintainers": ["siddharth7113"],
        "col_params": ["aptamer_col", "protein_col"],
    }

    def __init__(
        self, k=4, aptamer_col="aptamer", protein_col="protein", estimator=None
    ):
        self.k = k
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        self.estimator = estimator
        super().__init__()

    def _build_pipeline(self):
        pseaac = PSeAAC(
            lambda_val=30, weight=0.05, prop_indices=list(range(21)), group_props=3
        )
        features = ColumnTransformer(
            [
                ("aptamer", KMerFrequencies(k=self.k), [self.aptamer_col]),
                ("protein", pseaac, [self.protein_col]),
            ]
        )
        estimator = self.estimator
        if estimator is None:
            estimator = AptaNetClassifier()
        return Pipeline([("features", features), ("clf", clone(estimator))])

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"estimator": DummyClassifier()}
