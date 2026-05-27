__author__ = "Alleny244"
__all__ = ["plot_interaction_map"]

import numpy as np


def plot_interaction_map(
    interaction_map,
    aptamer_seq=None,
    protein_seq=None,
    title="Aptamer–Protein Interaction Map",
    cmap="viridis",
    ax=None,
):
    """
    Plot a 2D interaction heatmap from AptaTrans.

    Parameters
    ----------
    interaction_map : array-like of shape (apt_len, prot_len)
        Interaction scores between aptamer and protein positions.
        Accepts NumPy arrays or PyTorch tensors.
    aptamer_seq : str, optional
        Aptamer sequence for axis labels. If longer than the map
        dimension, it is truncated.
    protein_seq : str, optional
        Protein sequence for axis labels. If longer than the map
        dimension, it is truncated.
    title : str, default="Aptamer–Protein Interaction Map"
        Title for the plot.
    cmap : str, default="viridis"
        Matplotlib colormap name.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted interaction map.
    """
    import matplotlib.pyplot as plt

    if hasattr(interaction_map, "detach"):
        interaction_map = interaction_map.detach().cpu().numpy()

    interaction_map = np.asarray(interaction_map).squeeze()
    if interaction_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D interaction map, got shape {interaction_map.shape}."
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(interaction_map, aspect="auto", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    apt_len, prot_len = interaction_map.shape

    if aptamer_seq is not None:
        labels = list(aptamer_seq[:apt_len])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)

    if protein_seq is not None:
        labels = list(protein_seq[:prot_len])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)

    ax.set(
        xlabel="Protein Position",
        ylabel="Aptamer Position",
        title=title,
    )

    return ax
