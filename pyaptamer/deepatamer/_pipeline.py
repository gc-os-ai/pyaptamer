__author__ = "satvshr"
__all__ = ["DeepAptamerPipeline"]

import numpy as np
import torch

from pyaptamer.deepatamer._preprocessing import preprocess_seq_ohe, preprocess_seq_shape


class DeepAptamerPipeline:
    """
    DeepAptamer algorithm for aptamer–protein interaction prediction ..[1]

    This class encapsulates preprocessing (sequence one-hot encoding and DNAshape
    feature extraction) together with inference on a trained `DeepAptamerNN` model.
    It provides a `predict` method that accepts one or more DNA sequences and returns
    ranked binding affinity scores.

    References
    ----------
    .. [1] Yang X, Chan CH, Yao S, Chu HY, Lyu M, Chen Z, Xiao H, Ma Y, Yu S, Li F,
    Liu J, Wang L, Zhang Z, Zhang BT, Zhang L, Lu A, Wang Y, Zhang G, Yu Y.
    DeepAptamer: Advancing high-affinity aptamer discovery with a hybrid deep learning
    model. Mol Ther Nucleic Acids. 2024 Dec 21;36(1):102436.
    doi: 10.1016/j.omtn.2024.102436. PMID: 39897584; PMCID: PMC11787022.
    https://www.cell.com/molecular-therapy-family/nucleic-acids/pdf/S2162-2531(24)00323-8.pdf
    .. [2] deepDNAshape: a deep learning predictor for DNA shape features.
    https://github.com/JinsenLi/deepDNAshape/blob/main/LICENSE
    .. [3] DeepAptamer: a deep learning framework for aptamer design and binding
    prediction.
    https://github.com/YangX-BIDD/DeepAptamer

    Parameters
    ----------
    model : DeepAptamerNN
        A trained DeepAptamer neural network model.
    use_126_shape : bool, optional, default=True
        If True, use the trimmed 126-length DNAshape representation
        (MGW=31, HelT=32, ProT=31, Roll=32).
        If False, keep the full 138-length DeepDNAshape representation.
        (MGW=35, HelT=34, ProT=35, Roll=34).
    device : {"cpu", "cuda"}, default="cpu"
        Device to run inference on.

    Methods
    -------
    predict(seqs)
        Compute ranked binding affinity scores for one or more DNA
        sequences. Returns a list of dictionaries with each sequence
        and its predicted binding probability.
    """

    def __init__(self, model, use_126_shape=True, device="cpu"):
        self.model = model
        self.use_126_shape = use_126_shape
        self.device = device

    def predict(self, seqs):
        """
        Predict binding affinity scores for one or more sequences.

        Parameters
        ----------
        seqs : str or list of str
            DNA sequence(s), each length ≤ 35.

        Returns
        -------
        list of dict
            Ranked list of dictionaries, each with:
            {
                "seq": sequence string,
                "score": float (probability of binding, from [p_bind, p_not_bind])
            }
            Sorted from high to low by score.
        """
        if isinstance(seqs, str):
            seqs = [seqs]

        ohe_list, shape_list = [], []
        for seq in seqs:
            print(f"Processing sequence: {seq}")
            ohe_list.append(preprocess_seq_ohe(seq))
            shape_list.append(preprocess_seq_shape(seq))

        X_ohe = torch.tensor(
            np.array(ohe_list), dtype=torch.float32, device=self.device
        )
        X_shape = torch.tensor(
            np.array(shape_list), dtype=torch.float32, device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_ohe, X_shape)
            # convert to probabilities
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        # Take the "binding" probability
        bind_scores = probs[:, 0]

        # Create ranked output
        ranked = sorted(
            [
                {"seq": s, "score": float(sc)}
                for s, sc in zip(seqs, bind_scores, strict=False)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        return ranked
