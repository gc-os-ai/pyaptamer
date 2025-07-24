__author__ = ["nennomp"]
__all__ = ["Aptamer"]

import torch


class Aptamer:
    """Candidate aptamer evaluation for a given target protein.
    
    Examples
    --------
    from pyaptamer.experiment import Aptamer
    >>> experiment = Aptamer(target_encoded, target, model, device)
    >>> score = experiment.run(aptamer_candidate)
    """

    def __init__(
        self, 
        target_encoded: torch.Tensor, 
        target: str, 
        model: torch.nn.Module,
        device: torch.device, 
    ) -> None:
        """
        Parameters
        ----------
        target_encoded : torch.Tensor
            Encoded target sequence tensor.
        target : str, optional
            Target sequence string.
        model : torch.nn.Module
            Model to use for assigning scores.
        device : torch.device
            Device to run the model on.
        """
        self.target_encoded = target_encoded.to(device)
        self.target = target
        self.model = model
        self.device = device
    
    def _inputnames(self) -> list[str]:
        """Return the inputs of the experiment."""
        return ["aptamer_candidate"]

    @torch.no_grad()
    def run(self, aptamer_candidate: torch.Tensor) -> None:
        """Evaluate the given aptamer candidate.
        
        Parameters
        ----------
        aptamer_candidate : torch.Tensor
            The aptamer candidate to evaluate.
        
        Returns
        -------
        torch.Tensor
            The score assigned to the aptamer candidate.
        """
        self.model.eval()
        return self.model(
            aptamer_candidate.to(self.device), 
            self.target_encoded,
        )