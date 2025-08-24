"""Test suite for BaseSolver abstract class."""

__author__ = ["nennomp"]

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pyaptamer.base._base_solver import BaseSolver


class MockMLP(nn.Module):
    """Minimal MLP for testing."""

    def __init__(self, input_size: int = 8, hidden_size: int = 4, num_classes: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x1, x2):
        return self.layers(x1)


class TestSolver(BaseSolver):
    """Concrete implementation of BaseSolver for testing purposes."""

    def _compute_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy."""
        predictions = torch.argmax(outputs, dim=1)
        return (predictions == targets).float().mean().item()


class TestBaseSolver:
    """Tests for the BaseSolver abstract class."""

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
        train_data_apta = torch.randn(16, 8, device=device)
        train_data_prot = torch.randn(16, 8, device=device)
        train_labels = torch.randint(0, 2, (16,), device=device)
        test_data_apta = torch.randn(8, 8, device=device)
        test_data_prot = torch.randn(8, 8, device=device)
        test_labels = torch.randint(0, 2, (8,), device=device)

        train_loader = DataLoader(
            TensorDataset(train_data_apta, train_data_prot, train_labels), batch_size=4
        )
        test_loader = DataLoader(
            TensorDataset(test_data_apta, test_data_prot, test_labels), batch_size=4
        )

        return train_loader, test_loader

    @pytest.fixture
    def solver(self, device, mock_data):
        """Create basic TestSolver instance."""
        train_loader, test_loader = mock_data
        return TestSolver(
            device=device,
            model=MockMLP(),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            lr=0.001,
            weight_decay=0.0,
            momentum=0.9,
            betas=(0.9, 0.999),
        )

    def test_abstract_instantiation(self):
        """Check that BaseSolver cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseSolver(
                device=torch.device("cpu"),
                model=MockMLP(),
                train_dataloader=None,
                test_dataloader=None,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.AdamW,
                lr=0.001,
                weight_decay=0.0,
                momentum=0.9,
                betas=(0.9, 0.999),
            )

    def test_abstract_method_required(self):
        """Check that _compute_metric is abstract and must be implemented."""

        class IncompleteSolver(BaseSolver):
            pass  # Missing _compute_metric implementation

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteSolver(
                device=torch.device("cpu"),
                model=MockMLP(),
                train_dataloader=None,
                test_dataloader=None,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.AdamW,
                lr=0.001,
                weight_decay=0.0,
                momentum=0.9,
                betas=(0.9, 0.999),
            )

    def test_init_required_criterion(self, device, mock_data):
        """Check that criterion is now required."""
        train_loader, test_loader = mock_data

        # This should work with criterion provided
        solver = TestSolver(
            device=device,
            model=MockMLP(),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            lr=0.001,
            weight_decay=0.0,
            momentum=0.9,
            betas=(0.9, 0.999),
        )
        assert isinstance(solver.criterion, nn.CrossEntropyLoss)

    def test_init_defaults(self, device, mock_data):
        """Check BaseSolver initializes with correct defaults."""
        train_loader, test_loader = mock_data
        solver = TestSolver(
            device=device,
            model=MockMLP(),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            lr=0.001,
            weight_decay=0.0,
            momentum=0.9,
            betas=(0.9, 0.999),
        )

        assert isinstance(solver.optimizer, torch.optim.AdamW)  # Default is now AdamW
        assert isinstance(solver.criterion, nn.CrossEntropyLoss)
        assert solver.best_epoch == -1
        assert len(solver.history["train_loss"]) == 0
        assert len(solver.history["train_metric"]) == 0
        assert len(solver.history["test_loss"]) == 0
        assert len(solver.history["test_metric"]) == 0

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
        solver = TestSolver(
            device=device,
            model=MockMLP(),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer_class,
            lr=lr,
            weight_decay=0.0,
            momentum=0.9,
            betas=(0.9, 0.999),
        )

        assert isinstance(solver.optimizer, optimizer_class)
        assert solver.optimizer.param_groups[0]["lr"] == lr

    def test_init_unsupported_optimizer(self, device, mock_data):
        """Check ValueError for unsupported optimizer."""
        train_loader, test_loader = mock_data
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            TestSolver(
                device=device,
                model=MockMLP(),
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=torch.optim.RMSprop,
                lr=0.001,
                weight_decay=0.0,
                momentum=0.9,
                betas=(0.9, 0.999),
            )

    def test_compute_metric_implementation(self, solver):
        """Check that _compute_metric works correctly in concrete implementation."""
        # Create some mock outputs and targets
        outputs = torch.tensor(
            [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]
        )  # 3 samples, 2 classes
        targets = torch.tensor([1, 0, 1])  # Expected classes

        accuracy = solver._compute_metric(outputs, targets)

        # Expected: predictions are [1, 0, 1], targets are [1, 0, 1] -> 100% accuracy
        assert accuracy == 1.0

        # Test with wrong predictions
        outputs_wrong = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        accuracy_wrong = solver._compute_metric(outputs_wrong, targets)

        # Expected: predictions are [0, 1, 0], targets are [1, 0, 1] -> 0% accuracy
        assert accuracy_wrong == 0.0

    def test_evaluate(self, solver):
        """Check evaluate() returns loss and metric."""
        loss, metric = solver.evaluate(solver.test_dataloader, show_progress=False)

        assert isinstance(loss, float)
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0  # Accuracy should be between 0 and 1
        assert loss >= 0.0

    def test_train_single_epoch(self, solver):
        """Check train() works for one epoch."""
        history = solver.train(epochs=1, show_progress=False)

        assert len(history["train_loss"]) == 1
        assert len(history["test_loss"]) == 1
        assert len(history["train_metric"]) == 1
        assert len(history["test_metric"]) == 1
        assert solver.best_epoch == 0
        assert solver.best_params is not None

    @pytest.mark.parametrize(
        "monitor", ["test_loss", "test_metric"]
    )  # Updated from "test_accuracy"
    def test_train_monitor_options(self, solver, monitor):
        """Check train() works with different monitor metrics."""
        history = solver.train(epochs=1, monitor=monitor, show_progress=False)

        assert len(history["train_loss"]) == 1
        assert len(history["train_metric"]) == 1
        assert solver.best_epoch >= 0

    def test_train_invalid_monitor(self, solver):
        """Check ValueError for invalid monitor."""
        with pytest.raises(ValueError, match="Invalid monitor metric"):
            solver.train(epochs=1, monitor="invalid")

    @pytest.mark.parametrize("show_progress", [True, False])
    def test_train_show_progress(self, solver, show_progress):
        """Check train() works with different show_progress settings."""
        history = solver.train(epochs=1, show_progress=show_progress)

        assert len(history["train_loss"]) == 1
        assert len(history["test_loss"]) == 1

    def test_load_best_model_success(self, solver):
        """Check load_best_model() works after training."""
        solver.train(epochs=1, show_progress=False)

        # Modify model weights
        with torch.no_grad():
            for param in solver.model.parameters():
                param.data.fill_(999.0)

        # Load best model should restore the weights
        solver.load_best_model()

        # Check that weights were restored (not all 999.0)
        for param in solver.model.parameters():
            assert not torch.all(param.data == 999.0)

    def test_load_best_model_fail(self, solver):
        """Check load_best_model() fails when no training has occurred."""
        with pytest.raises(
            ValueError, match="No best model state available. Train the model first."
        ):
            solver.load_best_model()

    def test_reset_method(self, solver):
        """Check that _reset() properly initializes all attributes."""
        # Modify some attributes
        solver.best_epoch = 5
        solver.best_params = {"test": "value"}
        solver.history["train_loss"] = [1.0, 2.0]

        # Reset should restore defaults
        solver._reset()

        assert solver.best_epoch == -1
        assert solver.best_params is None
        assert len(solver.history["train_loss"]) == 0
        assert len(solver.history["train_metric"]) == 0
        assert len(solver.history["test_loss"]) == 0
        assert len(solver.history["test_metric"]) == 0

    def test_criterion_usage_in_training(self, solver):
        """Check that criterion is properly used during training."""
        # This test ensures the criterion is actually being called
        # by checking that training modifies model parameters
        initial_params = [param.clone() for param in solver.model.parameters()]

        solver.train(epochs=1, show_progress=False)

        # At least one parameter should have changed after training
        params_changed = any(
            not torch.equal(initial, current)
            for initial, current in zip(
                initial_params, solver.model.parameters(), strict=False
            )
        )
        assert params_changed, "Model parameters should change during training"

    def test_device_placement(self, device, mock_data):
        """Check that model is properly moved to specified device."""
        train_loader, test_loader = mock_data
        solver = TestSolver(
            device=device,
            model=MockMLP(),
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.AdamW,
            lr=0.001,
            weight_decay=0.0,
            momentum=0.9,
            betas=(0.9, 0.999),
        )

        # Check that model parameters are on the correct device
        for param in solver.model.parameters():
            assert param.device == device
