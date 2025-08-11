__author__ = ["nennomp"]
__all__ = ["Solver"]

import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Solver:
    """
    Class implementing training logic via backpropagation for neural networks, for
    classification tasks.

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
    optimizer : Type[Optimizer], optional, default=None
        The optimizer class to use for training. If None, Adam is used.
    criterion : nn.Module, optional, default=None
        The loss function to use. If None, CrossEntropyLoss is used.
    lr: float, optional, default=0.001
        Learning rate for the optimizer.
    weight_decay: float, optional, default=0.0
        Weight decay (L2 penalty) for the optimizer.
    momentum: float, optional, default=0.9
        Momentum coefficient for SGD optimizer. Only used if optimizer is SGD.
    betas: tuple[float, float], optional, default=(0.9, 0.999)
        Momentum coefficients for Adam and AdamW optimizers. Only used if optimizer is
        Adam or AdamW.

    Attributes
    ----------
    optimizer : Optimizer
        The optimizer initialized with the model parameters.
    criterion : nn.Module
        The loss function used for training.
    history : dict
        A dictionary to store training and test loss/accuracy history.
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
        optimizer: type[Optimizer] = None,
        criterion: nn.Module | None = None,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = self._init_optimizer(
            optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum, betas=betas
        )

        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion

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
        Optimizer
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
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }

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
        """Compute loss and accuracy on the specified dataloader in inference mode.

        Returns
        -------
        tuple[float, float]
            A tuple containing (loss, accuracy).
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, targets in tqdm(dataloader, disable=not show_progress):
            data, targets = data.to(self.device), targets.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * data.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

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
    ) -> dict[str, list]:
        """Train the model.

        Parameters
        ----------
        epochs : int, optional, default=100
            Number of epochs to train.
        monitor : str, optional, default="test_loss"
            Metric to monitor for best model selection. Options are 'test_loss' or
            'test_accuracy'.
        show_progress : bool, optional, default=True
            Whether to show training progress with tqdm.

        Returns
        -------
        dict[str, list]
            History of training and test loss/accuracy.

        Raises
        ------
        ValueError
            If monitor metric is not valid.
        """
        if monitor not in ["test_loss", "test_accuracy"]:
            raise ValueError(
                f"Invalid monitor metric: {monitor}. "
                "Options are 'test_loss' or 'test_accuracy'."
            )

        best_metric = float("inf") if monitor == "test_loss" else -float("inf")

        for epoch in range(epochs):
            # perform one training epoch
            self._run_epoch(show_progress=show_progress)

            # inference
            train_loss, train_acc = self.evaluate(
                self.train_dataloader, show_progress=show_progress
            )
            test_loss, test_acc = self.evaluate(
                self.test_dataloader, show_progress=show_progress
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_accuracy"].append(test_acc)

            # check whether model improved or not
            current_metric = test_loss if monitor == "test_loss" else test_acc
            has_improved = (
                monitor == "test_loss" and current_metric < best_metric
            ) or (monitor == "test_accuracy" and current_metric > best_metric)
            if has_improved:
                best_metric = current_metric
                self.best_epoch = epoch
                self.best_params = copy.deepcopy(self.model.state_dict())

            print(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}."
            )

        print(
            f"\nBest model found at epoch {self.best_epoch + 1} - "
            f"{monitor}: {best_metric:.4f}."
        )

        return self.history
