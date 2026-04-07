__author__ = ["tarun-227"]

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pyaptamer.utils import plot_interaction_map


class TestPlotInteractionMap:
    """Tests for the plot_interaction_map() function."""

    def test_basic_4d_input(self):
        """Test with standard (batch, 1, H, W) input."""
        imap = np.random.rand(1, 1, 10, 15)
        ax = plot_interaction_map(imap)
        assert ax is not None
        assert ax.get_title() == "Aptamer-Protein Interaction Map"

    def test_2d_input(self):
        """Test with already-squeezed (H, W) input."""
        imap = np.random.rand(10, 15)
        ax = plot_interaction_map(imap)
        assert ax is not None

    def test_with_sequence_labels(self):
        """Test that sequence labels appear on axes."""
        imap = np.random.rand(1, 1, 5, 8)
        candidate = "ACGUA"
        target = "DHRNENAI"
        ax = plot_interaction_map(imap, candidate=candidate, target=target)

        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert ytick_labels == list(candidate)
        assert xtick_labels == list(target)

    def test_custom_ax(self):
        """Test passing an existing axes object."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        imap = np.random.rand(1, 1, 5, 5)
        returned_ax = plot_interaction_map(imap, ax=ax)
        assert returned_ax is ax

    def test_custom_cmap(self):
        """Test with a custom colormap."""
        imap = np.random.rand(1, 1, 5, 5)
        ax = plot_interaction_map(imap, cmap="hot")
        assert ax is not None

    def test_torch_tensor_input(self):
        """Test that torch tensors are handled correctly."""
        torch = pytest.importorskip("torch")
        imap = torch.rand(1, 1, 5, 8)
        ax = plot_interaction_map(imap)
        assert ax is not None

    def test_invalid_shape_raises(self):
        """Test that a 3D array that can't squeeze to 2D raises ValueError."""
        imap = np.random.rand(2, 3, 5)
        with pytest.raises(ValueError, match="Expected a 2D interaction map"):
            plot_interaction_map(imap)

    def test_sequence_longer_than_map(self):
        """Test that labels are truncated to the map dimensions."""
        imap = np.random.rand(1, 1, 3, 4)
        candidate = "ACGUACGU"  # longer than 3
        target = "DHRNENAIQQQ"  # longer than 4
        ax = plot_interaction_map(imap, candidate=candidate, target=target)

        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(ytick_labels) == 3
        assert len(xtick_labels) == 4
