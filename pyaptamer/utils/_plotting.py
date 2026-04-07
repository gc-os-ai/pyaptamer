"""Plotting utilities for pyaptamer."""

__author__ = ["tarun-227"]
__all__ = ["plot_interaction_map"]

import numpy as np


def plot_interaction_map(
    interaction_map,
    candidate=None,
    target=None,
    ax=None,
    cmap="viridis",
    figsize=(10, 8),
    **kwargs,
):
    """Plot an aptamer-protein interaction map as a 2D heatmap.

    Takes the raw output of ``AptaTransPipeline.get_interaction_map()`` and
    renders it as a heatmap with optional sequence labels on the axes.

    Parameters
    ----------
    interaction_map : numpy.ndarray or torch.Tensor
        Interaction map tensor of shape ``(batch_size, 1, seq_len_apta,
        seq_len_prot)`` or ``(seq_len_apta, seq_len_prot)``.
    candidate : str or None, optional
        Aptamer candidate sequence. If provided, individual nucleotides are
        used as y-axis tick labels.
    target : str or None, optional
        Target protein sequence. If provided, individual residues are used
        as x-axis tick labels.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure and axes are created.
    cmap : str, optional, default="viridis"
        Matplotlib colormap name.
    figsize : tuple[int, int], optional, default=(10, 8)
        Figure size when creating a new figure (ignored if ``ax`` is given).
    **kwargs : dict
        Additional keyword arguments passed to ``ax.imshow()``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> from pyaptamer.utils import plot_interaction_map
    >>> imap = np.random.rand(1, 1, 5, 8)
    >>> ax = plot_interaction_map(imap, candidate="ACGUA", target="DHRNENAI")
    """
    import matplotlib.pyplot as plt

    imap = _squeeze_interaction_map(interaction_map)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(imap, aspect="auto", cmap=cmap, **kwargs)
    ax.figure.colorbar(im, ax=ax, label="Interaction intensity")

    ax.set_xlabel("Protein residues")
    ax.set_ylabel("Aptamer nucleotides")

    if target is not None:
        n_prot = imap.shape[1]
        labels = list(target[:n_prot])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)

    if candidate is not None:
        n_apta = imap.shape[0]
        labels = list(candidate[:n_apta])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)

    ax.set_title("Aptamer-Protein Interaction Map")

    return ax


def _squeeze_interaction_map(interaction_map):
    """Convert interaction map to a 2D numpy array.

    Parameters
    ----------
    interaction_map : numpy.ndarray or torch.Tensor
        Interaction map of shape ``(batch, 1, H, W)`` or ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        2D array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        If the array cannot be squeezed to 2D.
    """
    if hasattr(interaction_map, "detach"):
        interaction_map = interaction_map.cpu().detach().numpy()

    imap = np.squeeze(interaction_map)

    if imap.ndim != 2:
        raise ValueError(
            f"Expected a 2D interaction map after squeezing, got shape {imap.shape}. "
            "Input should be of shape (batch, 1, H, W) or (H, W)."
        )

    return imap
