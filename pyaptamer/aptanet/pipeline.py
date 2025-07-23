import numpy as np
import torch
import torch.optim as optim
from sklearn.pipeline import Pipeline

from pyaptamer.aptanet._model import FeatureSelector, SkorchAptaNet
from pyaptamer.pseaac import PSeAAC
from pyaptamer.utils._aptanet_utils import generate_kmer_vecs


# X is expected to be a list of tuples (aptamer_sequence, protein_sequence)
def pairs_to_features(X, k=4, pseaac_kwargs=None):
    pseaac_kwargs = {} if pseaac_kwargs is None else pseaac_kwargs
    pseaac = PSeAAC(**pseaac_kwargs)

    feats = []
    for aptamer_seq, protein_seq in X:
        kmer = generate_kmer_vecs(aptamer_seq, k=k)
        pseaac_vec = np.asarray(pseaac.transform(protein_seq))
        feats.append(np.concatenate([kmer, pseaac_vec]))
    return np.vstack(feats)


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
