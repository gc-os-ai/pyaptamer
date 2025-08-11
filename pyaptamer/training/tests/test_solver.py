"""Test suite for all solvers."""

__author__ = ["nennomp"]

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pyaptamer.training import Solver


class MockMLP(nn.Module):
    """Minimal MLP for testing."""

    def __init__(self, input_size: int = 8, hidden_size: int = 4, num_classes: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class TestSolver:
    """Tests for the Solver() class."""

    @pytest.fixture(
        params=[
            torch.device("cpu"),
            pytest.param(
                torch.device("cuda"),
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA not available"
                ),
            ),
        ]
    )
    def device(self, request):
        """Test device fixture for both CPU and CUDA."""
        return request.param

    @pytest.fixture
    def mock_data(self, device):
        """Create minimal synthetic dataset."""
        train_data = torch.randn(16, 8, device=device)
        train_labels = torch.randint(0, 2, (16,), device=device)
        test_data = torch.randn(8, 8, device=device)
        test_labels = torch.randint(0, 2, (8,), device=device)

        train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=4)
        test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=4)

        return train_loader, test_loader

    @pytest.fixture
    def solver(self, device, mock_data):
        """Create basic Solver instance."""
        train_loader, test_loader = mock_data
        return Solver(
            device=device,
            model=MockMLP(),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
        )

    def test_init_defaults(self, device, mock_data):
        """Check Solver initializes with defaults."""
        train_loader, test_loader = mock_data
        solver = Solver(
            device=device,
            model=MockMLP(),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
        )

        assert isinstance(solver.optimizer, torch.optim.Adam)
        assert isinstance(solver.criterion, nn.CrossEntropyLoss)
        assert solver.best_epoch == -1
        assert len(solver.history["train_loss"]) == 0

    @pytest.mark.parametrize(
        "optimizer_class, lr",
        [
            (torch.optim.SGD, 0.01),
            (torch.optim.Adam, 0.001),
            (torch.optim.AdamW, 0.0005),
        ],
    )
    def test_init_optimizers(self, device, mock_data, optimizer_class, lr):
        """Check different optimizer configurations."""
        train_loader, test_loader = mock_data
        solver = Solver(
            device=device,
            model=MockMLP(),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            optimizer=optimizer_class,
            lr=lr,
        )

        assert isinstance(solver.optimizer, optimizer_class)
        assert solver.optimizer.param_groups[0]["lr"] == lr

    def test_init_unsupported_optimizer(self, device, mock_data):
        """Check ValueError for unsupported optimizer."""
        train_loader, test_loader = mock_data
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            Solver(
                device=device,
                model=MockMLP(),
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=torch.optim.RMSprop,
            )

    def test_evaluate(self, solver):
        """Check evaluate() returns loss and accuracy."""
        loss, accuracy = solver.evaluate(solver.test_dataloader, show_progress=False)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert loss >= 0.0

    def test_train_single_epoch(self, solver):
        """Check train() works for one epoch."""
        history = solver.train(epochs=1, show_progress=False)

        assert len(history["train_loss"]) == 1
        assert len(history["test_loss"]) == 1
        assert solver.best_epoch == 0
        assert solver.best_params is not None

    @pytest.mark.parametrize("monitor", ["test_loss", "test_accuracy"])
    def test_train_monitor_options(self, solver, monitor):
        """Check train() works with different monitor metrics."""
        history = solver.train(epochs=1, monitor=monitor, show_progress=False)

        assert len(history["train_loss"]) == 1
        assert solver.best_epoch >= 0

    def test_train_invalid_monitor(self, solver):
        """Check ValueError for invalid monitor."""
        with pytest.raises(ValueError, match="Invalid monitor metric"):
            solver.train(epochs=1, monitor="invalid")

    @pytest.mark.parametrize("show_progress", [True, False])
    def test_train(self, solver, show_progress):
        """Check load_best_model() after training."""
        solver.train(epochs=1, show_progress=show_progress)
        solver.load_best_model()

    def test_load_best_model_fail(self, solver):
        """Check the loading best model fails when it is None."""
        with pytest.raises(
            ValueError, match="No best model state available. Train the model first."
        ):
            solver.load_best_model()
