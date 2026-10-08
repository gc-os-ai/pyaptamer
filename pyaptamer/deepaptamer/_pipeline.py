__author__ = ["satvshr", "geetu040"]
__all__ = ["DeepAptamerPipeline"]

import pandas as pd
from skbase.base import BaseEstimator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from pyaptamer.data import MoleculeLoader
from pyaptamer.deepaptamer._classifier import DeepAptamerClassifier
from pyaptamer.deepaptamer._preprocessing import DeepAptamerFeatures


class DeepAptamerPipeline(BaseEstimator):
    """
    DeepAptamer algorithm for aptamer binding prediction [1]_

    Implements DeepAptamer, a hybrid deep learning model that combines the one-hot
    encoded aptamer sequence with its predicted DNA shape (MGW, HelT, ProT, Roll)
    to predict whether an aptamer binds its target (binary classification).

    The pipeline takes a MoleculeLoader or a DataFrame with an aptamer column. The
    aptamers are encoded with `DeepAptamerFeatures` and passed to the estimator.

    Parameters
    ----------
    seq_len : int, optional, default=35
        Length the aptamer sequences are padded to. Longer sequences raise an error.
    full_dna_shape : bool, optional, default=False
        If True, keep the full `DeepDNAShape` output (138 values for a 35-mer).
        If False, drop the edge positions that are NA in `DNAshapeR`, as in the
        DeepAptamer paper (126 values for a 35-mer).
    aptamer_col : str, optional, default="aptamer"
        Name of the column holding aptamer sequences.
    estimator : sklearn-compatible estimator or None, default=None
        Estimator applied to the features. If None, uses `DeepAptamerClassifier`
        with the same ``seq_len``. A custom `DeepAptamerClassifier` must use the
        same ``seq_len`` as the pipeline.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Steps ``features`` (a ``ColumnTransformer``) and ``clf``.

    References
    ----------
    .. [1] Yang X, Chan CH, Yao S, Chu HY, Lyu M, Chen Z, Xiao H, Ma Y, Yu S, Li F,
       Liu J, Wang L, Zhang Z, Zhang BT, Zhang L, Lu A, Wang Y, Zhang G, Yu Y.
       DeepAptamer: Advancing high-affinity aptamer discovery with a hybrid deep
       learning model. Mol Ther Nucleic Acids. 2024 Dec 21;36(1):102436.
       doi: 10.1016/j.omtn.2024.102436. PMID: 39897584; PMCID: PMC11787022.
       https://www.cell.com/molecular-therapy-family/nucleic-acids/pdf/S2162-2531(24)00323-8.pdf
    .. [2] deepDNAshape: a deep learning predictor for DNA shape features.
       https://github.com/JinsenLi/deepDNAshape/blob/main/LICENSE
    .. [3] DeepAptamer: a deep learning framework for aptamer design and binding
       prediction.
       https://github.com/YangX-BIDD/DeepAptamer

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.deepaptamer import DeepAptamerClassifier, DeepAptamerPipeline
    >>> aptamers = [
    ...     "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCC",
    ...     "TGCATGCTAGCTAGCTAGCTAGCTAGCTAGCGCTA",
    ... ]
    >>> X_train = MoleculeLoader(data={"aptamer": aptamers * 10})
    >>> y_train = np.array([0, 1] * 10)
    >>> pipe = DeepAptamerPipeline(estimator=DeepAptamerClassifier(max_epochs=2))
    >>> pipe.fit(X_train, y_train)  # doctest: +ELLIPSIS
    DeepAptamerPipeline(...)
    >>> preds = pipe.predict(X_train)
    >>> proba = pipe.predict_proba(X_train)
    """

    def __init__(
        self, seq_len=35, full_dna_shape=False, aptamer_col="aptamer", estimator=None
    ):
        self.seq_len = seq_len
        self.full_dna_shape = full_dna_shape
        self.aptamer_col = aptamer_col
        self.estimator = estimator
        super().__init__()

    def _build_pipeline(self):
        features = ColumnTransformer(
            [
                (
                    "aptamer",
                    DeepAptamerFeatures(
                        seq_len=self.seq_len, full_dna_shape=self.full_dna_shape
                    ),
                    [self.aptamer_col],
                ),
            ]
        )
        estimator = self.estimator or DeepAptamerClassifier(seq_len=self.seq_len)
        return Pipeline([("features", features), ("clf", clone(estimator))])

    @staticmethod
    def _to_frame(X):
        if isinstance(X, MoleculeLoader):
            return X.to_dataframe()
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a MoleculeLoader instance or a pandas DataFrame. "
                f"Got {type(X)} instead."
            )
        return X

    def fit(self, X, y):
        """
        Fit the pipeline on training data.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Training data with the aptamer column.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : object
            Fitted pipeline.
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(self._to_frame(X), y)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for the aptamers in `X`.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Data with the aptamer column.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probability estimates for each class, in the order of the estimator's
            ``classes_``.
        """
        self.check_is_fitted(method_name="predict_proba")
        return self.pipeline_.predict_proba(self._to_frame(X))

    def predict(self, X):
        """
        Predict binary class labels for the aptamers in `X`.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Data with the aptamer column.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self.check_is_fitted(method_name="predict")
        return self.pipeline_.predict(self._to_frame(X))
