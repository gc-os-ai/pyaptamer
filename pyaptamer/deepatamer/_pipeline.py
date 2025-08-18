import numpy as np
import torch

from pyaptamer.deepatamer._model import DeepAptamer
from pyaptamer.deepatamer._preprocessing import preprocess_seq_ohe, preprocess_seq_shape


class DeepAptamerPipeline:
    def __init__(self, use_126_shape=True, device="cpu"):
        """
        Initializes the DeepAptamer pipeline.

        Parameters
        ----------
        use_126_shape : bool, optional
            If True, uses the 126-length shape vector (matching DNAshape).
            If False, uses the 138-length vector from DeepDNAshape.
        device : {"cpu", "cuda"}, optional
            Device for running inference.
        """
        self.model = DeepAptamer().to(device)
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

        # Preprocess all sequences
        ohe_list, shape_list = [], []
        for seq in seqs:
            ohe_list.append(preprocess_seq_ohe(seq))
            shape_list.append(preprocess_seq_shape(seq))

        # Convert to tensors
        x_ohe = torch.tensor(
            np.array(ohe_list), dtype=torch.float32, device=self.device
        )
        x_shape = torch.tensor(
            np.array(shape_list), dtype=torch.float32, device=self.device
        )

        # Run model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_ohe, x_shape)  # shape (batch_size, 2)
            probs = (
                torch.softmax(outputs, dim=1).cpu().numpy()
            )  # convert to probabilities

        # Take the "binding" probability (index 0 here — adjust if flipped)
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
