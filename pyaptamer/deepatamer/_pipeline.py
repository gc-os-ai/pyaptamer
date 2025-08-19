__author__ = "satvshr"
__all__ = ["DeepAptamerPipeline"]

import numpy as np
import torch

from pyaptamer.deepatamer._preprocessing import preprocess_seq_ohe, preprocess_seq_shape


class DeepAptamerPipeline:
    """
    Initializes the DeepAptamer pipeline.

    Parameters
    ----------
    model : DeepAptamer
        The pre-trained DeepAptamer model.
    use_126_shape : bool, optional
        If True, uses the 126-length shape vector (matching DNAshape).
        If False, uses the 138-length vector from DeepDNAshape.
    device : {"cpu", "cuda"}, optional
        Device for running inference.
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
