import numpy as np
import torch
import torch.optim as optim
from sklearn.pipeline import Pipeline

from pyaptamer.aptanet import FeatureSelector, SkorchAptaNet
from pyaptamer.pseaac import PSeAAC
from pyaptamer.utils._aptanet_utils import generate_kmer_vecs


def pairs_to_features(X, k=4, pseaac_kwargs=None):
    """
    Convert a list of (aptamer_sequence, protein_sequence) pairs into feature vectors.

    This function generates feature vectors for each (aptamer, protein) pair using:
    - k-mer representation of the aptamer sequence
    - Pseudo amino acid composition (PSeAAC) representation of the protein sequence

    Parameters
    ----------
    X : list of tuple of str
        A list where each element is a tuple `(aptamer_sequence, protein_sequence)`.
        `aptamer_sequence` should be a string of nucleotides, and `protein_sequence`
        should be a string of amino acids.

    k : int, optional
        The k-mer size used to generate the k-mer vector from the aptamer sequence.
        Default is 4.

    pseaac_kwargs : dict, optional
        Optional keyword arguments to pass to the `PSeAAC` transformer.
        If not provided, default parameters are used.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row corresponds to the concatenated feature vector
        for a given (aptamer, protein) pair.
    """
    pseaac_kwargs = {} if pseaac_kwargs is None else pseaac_kwargs
    pseaac = PSeAAC(**pseaac_kwargs)

    feats = []
    for aptamer_seq, protein_seq in X:
        kmer = generate_kmer_vecs(aptamer_seq, k=k)
        pseaac_vec = np.asarray(pseaac.transform(protein_seq))
        feats.append(np.concatenate([kmer, pseaac_vec]))

    # Ensure float32 for PyTorch compatibility
    return np.vstack(feats).astype(np.float32)


net = SkorchAptaNet(
    module__hidden_dim=128,
    module__n_hidden=7,
    module__dropout=0.3,
    max_epochs=200,
    lr=1.4e-4,
    batch_size=310,
    optimizer=optim.RMSprop,
    device="cuda" if torch.cuda.is_available() else "cpu",
    threshold=0.5,
    verbose=1,
)

pipe = Pipeline(
    [("features", pairs_to_features), ("select", FeatureSelector()), ("clf", net)]
)
