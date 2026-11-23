__author__ = ["satvshr", "geetu040"]
__all__ = ["DeepAptamerPipeline"]

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from pyaptamer.base import BasePipelineDelegator
from pyaptamer.deepaptamer._classifier import DeepAptamerClassifier
from pyaptamer.deepaptamer._preprocessing import DeepAptamerFeatures


class DeepAptamerPipeline(BasePipelineDelegator):
    """
    DeepAptamer algorithm for aptamer binding prediction [1]_

    Implements DeepAptamer, a hybrid deep learning model that combines the one-hot
    encoded aptamer sequence with its predicted DNA shape (MGW, HelT, ProT, Roll)
    to predict whether an aptamer binds its target (binary classification).

    This follows the original implementation [2]_.

    The pipeline takes a MoleculeLoader or a DataFrame with an aptamer column. The
    aptamers are encoded with `DeepAptamerFeatures` and passed to the estimator.

    Only DNA aptamers are supported. Sequences must be uppercase and contain only
    A, C, G, T and N. Other characters, including U in RNA, raise a ValueError.

    Parameters
    ----------
    seq_len : int, optional, default=35
        Length the aptamer sequences are padded to. Longer sequences raise an error.
        Must be at least 34, or 31 if ``full_dna_shape`` is True.
    full_dna_shape : bool, optional, default=False
        If True, keep the full `DeepDNAShape` output (138 values for a 35-mer).
        If False, drop the edge positions that are NA in `DNAshapeR`, as in the
        DeepAptamer paper (126 values for a 35-mer).
    aptamer_col : str, optional, default="aptamer"
        Name of the column holding aptamer sequences.
    estimator : sklearn-compatible estimator or None, default=None
        Estimator applied to the features. If None, uses `DeepAptamerClassifier`.
        If the estimator has a ``seq_len`` parameter, it is set to the pipeline's
        ``seq_len``.

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
    .. [2] DeepAptamer original implementation.
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

    _tags = {
        "authors": ["satvshr", "geetu040"],
        "maintainers": ["siddharth7113"],
        "col_params": ["aptamer_col"],
    }

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
        if self.estimator is None:
            estimator = DeepAptamerClassifier()
        else:
            estimator = clone(self.estimator)
        if "seq_len" in estimator.get_params():
            estimator.set_params(seq_len=self.seq_len)
        return Pipeline([("features", features), ("clf", estimator)])

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"estimator": DummyClassifier()}
