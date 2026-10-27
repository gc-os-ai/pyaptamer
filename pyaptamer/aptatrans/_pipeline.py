"""AptaTrans pipeline for aptamer-protein interaction prediction and candidate
aptamer generation.
"""

__author__ = ["nennomp", "siddharth7113"]
__all__ = ["AptaTransPipeline"]

import pandas as pd
from skbase.base import BaseEstimator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from pyaptamer.aptatrans._classifier import AptaTransClassifier
from pyaptamer.data import MoleculeLoader
from pyaptamer.datasets import load_csv_dataset
from pyaptamer.experiments import AptamerEvalAptaTrans
from pyaptamer.mcts import MCTS
from pyaptamer.transformations.compose import ApplyToCols
from pyaptamer.transformations.encode import GreedyEncoder, KMerIndexEncoder
from pyaptamer.transformations.transform import DNAToRNA
from pyaptamer.utils._base import filter_words


class AptaTransPipeline(BaseEstimator):
    """AptaTrans pipeline for aptamer-protein interaction, by Shin et al. [1]_.

    AptaTrans predicts whether an aptamer binds a protein from the two
    sequences. The aptamer is converted to RNA with ``DNAToRNA`` and encoded
    with ``KMerIndexEncoder``, the ids of its overlapping triplets over
    ``ACGUN``. The protein is encoded with ``GreedyEncoder``, a greedy longest
    match against the protein words. The two id blocks are concatenated and
    passed to ``AptaTransClassifier``.

    A fitted pipeline also generates candidate aptamers for a target protein
    with ``generate``. It drives ``pyaptamer.mcts.MCTS`` [2]_ with
    ``pyaptamer.experiments.AptamerEvalAptaTrans`` as the scoring function.

    The pipeline takes a ``MoleculeLoader`` or a ``DataFrame`` with an
    aptamer column and a protein column. Other columns are ignored.

    Original implementation: https://github.com/PNUMLB/AptaTrans.

    Parameters
    ----------
    prot_words : dict[str, float] or None, default=None
        Protein words mapped to their frequency. Words with a frequency above
        the mean are kept and get the ids 1, 2, and so on. They should come
        from the dataset used to pretrain the protein encoder. If None, uses
        the bundled ``protein_word_freq`` dataset of the paper.
    apta_max_len : int, default=275
        Number of triplet ids per aptamer.
    prot_max_len : int, default=867
        Number of word ids per protein.
    aptamer_col : str, default="aptamer"
        Name of the column holding aptamer sequences.
    protein_col : str, default="protein"
        Name of the column holding protein sequences.
    estimator : AptaTransClassifier or None, default=None
        Classifier applied to the token ids. Its ``apta_max_len`` is set to
        the ``apta_max_len`` of this pipeline. If None, uses
        ``AptaTransClassifier()``.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Steps ``to_rna``, ``features`` (a ``ColumnTransformer``) and ``clf``.

    References
    ----------
    .. [1] Shin, Incheol, et al. "AptaTrans: a deep neural network for predicting
       aptamer-protein interaction using pretrained encoders." BMC bioinformatics
       24.1 (2023): 447.
    .. [2] Lee, Gwangho, et al. "Predicting aptamer sequences that interact with
       target proteins using an aptamer-protein interaction classifier and a Monte
       Carlo tree search approach." PloS one 16.6 (2021): e0253760.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.aptatrans import AptaTransClassifier, AptaTransPipeline
    >>> from pyaptamer.data import MoleculeLoader
    >>> aptamer = "AGCTTAGCGTACAGCTTAAAAGGGTTTCCCCTGCCCGCGTAC"
    >>> protein = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    >>> X = MoleculeLoader(data={"aptamer": [aptamer] * 8, "protein": [protein] * 8})
    >>> y = np.array([0, 1] * 4)
    >>> clf = AptaTransClassifier(
    ...     in_dim=8,
    ...     n_heads=2,
    ...     n_encoder_layers=1,
    ...     conv_layers=[1, 1, 1],
    ...     max_epochs=1,
    ...     random_state=0,
    ... )
    >>> pipe = AptaTransPipeline(apta_max_len=16, prot_max_len=16, estimator=clf)
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    AptaTransPipeline(...)
    >>> proba = pipe.predict_proba(X)
    >>> labels = pipe.predict(X)
    >>> candidates = pipe.generate(protein, n_candidates=1, depth=5, n_iterations=2)
    """

    def __init__(
        self,
        prot_words=None,
        apta_max_len=275,
        prot_max_len=867,
        aptamer_col="aptamer",
        protein_col="protein",
        estimator=None,
    ):
        self.prot_words = prot_words
        self.apta_max_len = apta_max_len
        self.prot_max_len = prot_max_len
        self.aptamer_col = aptamer_col
        self.protein_col = protein_col
        self.estimator = estimator
        super().__init__()

    def _build_pipeline(self):
        prot_words = self.prot_words
        if prot_words is None:
            freq = load_csv_dataset(
                name="protein_word_freq", keep_default_na=False, na_values=["_"]
            )
            prot_words = freq.set_index("seq")["freq"].to_dict()

        features = ColumnTransformer(
            [
                (
                    "aptamer",
                    KMerIndexEncoder(k=3, alphabet="ACGUN", max_len=self.apta_max_len),
                    [self.aptamer_col],
                ),
                (
                    "protein",
                    GreedyEncoder(
                        words=filter_words(prot_words), max_len=self.prot_max_len
                    ),
                    [self.protein_col],
                ),
            ]
        )
        if self.estimator is None:
            estimator = AptaTransClassifier()
        else:
            estimator = clone(self.estimator)
        estimator.set_params(apta_max_len=self.apta_max_len)
        return Pipeline(
            [
                ("to_rna", ApplyToCols(DNAToRNA(), cols=self.aptamer_col)),
                ("features", features),
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
                "Pass the names as AptaTransPipeline(aptamer_col=..., "
                "protein_col=...)."
            )
        return X[columns]

    def fit(self, X, y):
        """Encode the pairs and train the classifier.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Aptamer and protein sequences, one pair per row.
        y : array-like of shape (n_samples,)
            Binary labels, 1 if the pair interacts.

        Returns
        -------
        self : AptaTransPipeline
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(self._to_frame(X), y)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        """Predict class probabilities for each pair.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Aptamer and protein sequences, one pair per row.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probability of each class, in the order of ``classes_`` of the
            classifier.
        """
        self.check_is_fitted(method_name="predict_proba")
        return self.pipeline_.predict_proba(self._to_frame(X))

    def predict(self, X):
        """Predict the class label of each pair.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Aptamer and protein sequences, one pair per row.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self.check_is_fitted(method_name="predict")
        return self.pipeline_.predict(self._to_frame(X))

    def interaction_map(self, X):
        """Compute the aptamer-protein interaction map of each pair.

        Parameters
        ----------
        X : MoleculeLoader or pandas.DataFrame
            Aptamer and protein sequences, one pair per row.

        Returns
        -------
        ndarray of shape (n_samples, 1, apta_len, prot_len)
            Interaction map between the aptamer and protein positions.
            ``apta_len`` is ``apta_max_len`` and ``prot_len`` is
            ``prot_max_len``, each at least 6.
        """
        self.check_is_fitted(method_name="interaction_map")
        tokens = self.pipeline_[:-1].transform(self._to_frame(X))
        return self.pipeline_[-1].interaction_map(tokens)

    def generate(self, target, n_candidates=10, depth=20, n_iterations=1000):
        """Generate candidate aptamers for a target protein.

        Each candidate is found by one run of Monte Carlo Tree Search, scored
        by this pipeline. Runs continue until ``n_candidates`` distinct
        candidates are found.

        Parameters
        ----------
        target : str
            Amino-acid sequence of the target protein.
        n_candidates : int, default=10
            Number of distinct candidates to return.
        depth : int, default=20
            Depth of the search tree. Also the number of nucleotides of each
            candidate. Must be at least 3, the length of one triplet.
        n_iterations : int, default=1000
            Number of search iterations per nucleotide.

        Returns
        -------
        list of tuple[str, str, float]
            ``(candidate, sequence, score)`` per candidate, highest score
            first. ``sequence`` is the raw search sequence with direction
            markers, before reconstruction.
        """
        self.check_is_fitted(method_name="generate")
        if depth < 3:
            raise ValueError(f"depth must be at least 3, got {depth}.")

        experiment = AptamerEvalAptaTrans(target=target, pipeline=self)
        mcts = MCTS(experiment=experiment, depth=depth, n_iterations=n_iterations)

        candidates = {}
        while len(candidates) < n_candidates:
            result = mcts.run(verbose=False)
            candidate = result["candidate"]
            if candidate not in candidates:
                candidates[candidate] = (
                    candidate,
                    result["sequence"],
                    float(result["score"]),
                )

        return sorted(candidates.values(), key=lambda c: c[2], reverse=True)
