# TODO: to remove if we end up refactoring to Pytorch Lightning

__author__ = ["nennomp"]
__all__ = ["BaseSolver"]

import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseSolver(ABC):
    """
    Abstract base class for training neural networks via backpropagation.

    This class defines the common structure and interface for training PyTorch models.
    Subclasses must implement the abstract methods for computing metrics and loss.

    Parameters
    ----------
    device : torch.device
        The device to use for training.
    model : nn.Module
        The model to be trained.
    train_dataloader : DataLoader
        Dataloader for training data.
    test_dataloader : DataLoader
        Dataloader for test data.
    criterion : torch.nn.modules.loss._Loss
        The loss function to use for training.
    optimizer : torch.optim.Optimizer,
        The optimizer class to use for training. If None, AdamW is used.
    lr: float
        Learning rate for the optimizer.
    weight_decay: float
        Weight decay (L2 penalty) for the optimizer.
    momentum: float
        Momentum coefficient for SGD optimizer. Only used if optimizer is SGD.
    betas: tuple[float, float]
        Momentum coefficients for Adam and AdamW optimizers.

    Attributes
    ----------
    criterion : torch.nn.modules.loss._Loss
        The loss function used for training.
    optimizer : torch.optim.Optimizer
        The optimizer initialized with the model parameters.
    history : dict
        A dictionary to store training and test loss history, and training and test
        target metric.
    best_params : dict
        State dict of the model with best validation performance.
    best_epoch : int
        Epoch number where the best model was found.
    """

    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        criterion,
        optimizer,
        lr: float,
        weight_decay: float,
        momentum: float,
        betas: tuple[float, float],
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.criterion = criterion

        self.optimizer = self._init_optimizer(
            optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum, betas=betas
        )

        self._reset()

    def _init_optimizer(
        self,
        optimizer: type[Optimizer],
        lr: float,
        weight_decay: float,
        momentum: float,
        betas: tuple[float, float],
    ) -> Optimizer:
        """Initialize the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer initialized with `self.model` parameters.

        Raises
        ------
        ValueError
            If the optimizer is not one of the supported types (SGD, Adam, AdamW).
        """
        if optimizer is None:
            optimizer = torch.optim.AdamW

        match optimizer:
            case torch.optim.SGD:
                return optimizer(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                )
            case torch.optim.Adam | torch.optim.AdamW:
                return optimizer(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                )
            case _:
                raise ValueError(
                    f"Unsupported optimizer: {optimizer.__name__}. "
                    "Options are 'SGD', 'Adam', and 'AdamW'."
                )

    def _reset(self) -> None:
        """Reset the model and optimizer states."""
        self.best_epoch = -1
        self.best_params = None

        self.model.train()
        self.optimizer.zero_grad()

        self.history = {
            "train_loss": [],
            "train_metric": [],
            "test_loss": [],
            "test_metric": [],
        }

    @abstractmethod
    def _compute_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute a metric (e.g., accuracy, F1, etc.) from model outputs and targets.

        This method should be implemented by subclasses to define task-specific
        metric computation.

        Parameters
        ----------
        outputs : torch.Tensor
            Model outputs.
        targets : torch.Tensor
            Ground truth labels.

        Returns
        -------
        float
            Computed metric value.
        """
        pass

    def _run_epoch(self, show_progress: bool) -> None:
        """Run a single training epoch."""
        self.model.train()

        for data, targets in tqdm(self.train_dataloader, disable=not show_progress):
            data, targets = data.to(self.device), targets.to(self.device)

            # forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # backward pass
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        show_progress: bool,
    ) -> tuple[float, float]:
        """Compute loss and metric on the specified dataloader in inference mode.

        Returns
        -------
        Tuple[float, float]
            A tuple containing (average_loss, metric).
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_targets = []

        for data, targets in tqdm(dataloader, disable=not show_progress):
            data, targets = data.to(self.device), targets.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * data.size(0)
            total_samples += targets.size(0)

            # Collect outputs and targets for metric computation
            all_outputs.append(outputs)
            all_targets.append(targets)

        avg_loss = total_loss / total_samples

        # Compute metric on all outputs/targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metric = self._compute_metric(all_outputs, all_targets)

        return avg_loss, metric

    def load_best_model(self) -> None:
        """Load the best model state found during training.

        Raises
        ------
        ValueError
            If no best model state is available.
        """
        if self.best_params is None:
            raise ValueError("No best model state available. Train the model first.")

        self.model.load_state_dict(self.best_params)

    def train(
        self, epochs: int = 100, monitor: str = "test_loss", show_progress: bool = True
    ) -> dict[str, list[float]]:
        """Train the model.

        Parameters
        ----------
        epochs : int, optional, default=100
            Number of epochs to train.
        monitor : str, optional, default="test_loss"
            Metric to monitor for best model selection. Options are 'test_loss' or
            'test_metric'.
        show_progress : bool, optional, default=True
            Whether to show training progress with tqdm.

        Returns
        -------
        Dict[str, List[float]]
            History of training and test loss and metric.

        Raises
        ------
        ValueError
            If monitor metric is not valid.
        """
        if monitor not in ["test_loss", "test_metric"]:
            raise ValueError(
                f"Invalid monitor metric: {monitor}. "
                "Options are 'test_loss' or 'test_metric'."
            )

        best_metric = float("inf") if monitor == "test_loss" else -float("inf")

        for epoch in range(epochs):
            # perform one training epoch
            self._run_epoch(show_progress=show_progress)

            # inference
            train_loss, train_metric = self.evaluate(
                self.train_dataloader, show_progress=show_progress
            )
            test_loss, test_metric = self.evaluate(
                self.test_dataloader, show_progress=show_progress
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_metric"].append(train_metric)
            self.history["test_loss"].append(test_loss)
            self.history["test_metric"].append(test_metric)

            # check whether model improved or not
            current_metric = test_loss if monitor == "test_loss" else test_metric
            has_improved = (
                monitor == "test_loss" and current_metric < best_metric
            ) or (monitor == "test_metric" and current_metric > best_metric)

            if has_improved:
                best_metric = current_metric
                self.best_epoch = epoch
                self.best_params = copy.deepcopy(self.model.state_dict())

            print(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Metric: {test_metric:.4f}."
            )

        print(
            f"\nBest model found at epoch {self.best_epoch + 1} - "
            f"{monitor}: {best_metric:.4f}."
        )

        return self.history
