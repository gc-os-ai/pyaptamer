"""RaptGen end-to-end generator on raw sequences"""

__author__ = ["NoorMajdoub"]
__all__ = ["RaptGenGenerator"]

from skbase.base import BaseEstimator
from sklearn.base import clone
from sklearn.utils import InputTags, Tags, TargetTags, TransformerTags

from pyaptamer.raptgen._estimator import RaptGenModel
from pyaptamer.transformations.encode import SequenceIndexEncoder
from pyaptamer.utils._check_x import check_X


class RaptGenGenerator(BaseEstimator):
    """RaptGen algorithm for unsupervised aptamer sequence generation [1]_.

    Takes a MoleculeLoader or a DataFrame with a column of fixed-length aptamer
    sequences. The sequences are encoded as integer indices with
    `SequenceIndexEncoder` and passed to the estimator, `RaptGenModel` by
    default, which trains the variational autoencoder. `transform` maps
    sequences to points in the latent space and `inverse_transform` generates
    sequences from latent points.

    Parameters
    ----------
    aptamer_col : str, optional, default="aptamer"
        Name of the column holding the aptamer sequences.
    estimator : RaptGenModel or None, optional, default=None
        Estimator fitted on the encoded sequences. If None, uses
        `RaptGenModel()` with its default parameters.

    Attributes
    ----------
    encoder_ : SequenceIndexEncoder
        The fitted sequence encoder.
    estimator_ : RaptGenModel
        The fitted estimator.

    References
    ----------
    .. [1] Iwano, Natsuki, et al. "Generative aptamer discovery using RaptGen."
    Nature Computational Science 2.6 (2022): 378-386.

    Examples
    --------
    >>> from pyaptamer.data import MoleculeLoader
    >>> from pyaptamer.raptgen import RaptGenGenerator, RaptGenModel
    >>> aptamers = ["ATGCATGCATGC", "GGCATTACGTAC", "TTGACCGTAGCA", "CCATGGTACGAT"]
    >>> X = MoleculeLoader(data={"aptamer": aptamers})
    >>> model = RaptGenModel(motif_len=5, max_epochs=1, random_state=0)
    >>> gen = RaptGenGenerator(estimator=model)
    >>> gen.fit(X)  # doctest: +ELLIPSIS
    RaptGenGenerator(...)
    >>> Z = gen.transform(X)
    >>> Z.shape
    (4, 10)
    >>> len(gen.inverse_transform(Z[:2]))
    2
    """

    def __init__(self, aptamer_col="aptamer", estimator=None):
        self.aptamer_col = aptamer_col
        self.estimator = estimator
        super().__init__()

    def _check_X(self, X):  # noqa: N802
        cols = {"aptamer_col": self.aptamer_col}
        return check_X(X, cols=cols, owner=type(self).__name__)

    def fit(self, X, y=None):
        """Train the model on aptamer sequences.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Data with the aptamer column. All sequences must have the same
            length and contain only A, T, G, C or U.
        y : ignored
            Present for API compatibility. RaptGen is unsupervised.

        Returns
        -------
        self : RaptGenGenerator
            The fitted generator.
        """
        self.encoder_ = SequenceIndexEncoder()
        X_idx = self.encoder_.fit_transform(self._check_X(X))
        self.estimator_ = clone(self.estimator) if self.estimator else RaptGenModel()
        self.estimator_.fit(X_idx.to_numpy())
        self._is_fitted = True
        return self

    def transform(self, X):
        """Encode aptamer sequences into latent-space points.

        Parameters
        ----------
        X : MoleculeLoader or pd.DataFrame
            Data with the aptamer column.

        Returns
        -------
        ndarray of shape (n_samples, embed_size)
            Latent-space coordinates.
        """
        self.check_is_fitted(method_name="transform")
        X_idx = self.encoder_.transform(self._check_X(X))
        return self.estimator_.transform(X_idx.to_numpy())

    def inverse_transform(self, Z, most_likely=True):
        """Generate sequences from latent-space points.

        Parameters
        ----------
        Z : array of shape (n_points, embed_size)
            Latent-space points.
        most_likely : bool, optional, default=True
            Passed to `RaptGenModel.inverse_transform`.

        Returns
        -------
        list of str
            One A/T/G/C sequence per point in `Z`.
        """
        self.check_is_fitted(method_name="inverse_transform")
        return self.estimator_.inverse_transform(Z, most_likely=most_likely)

    def __sklearn_tags__(self):
        """Return the scikit-learn tags of an unsupervised transformer."""
        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(preserves_dtype=[]),
            input_tags=InputTags(two_d_array=True, string=True),
        )
