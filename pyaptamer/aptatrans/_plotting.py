"""Plotting utilities for AptaTrans interaction maps."""

__author__ = ["tarun-227"]
__all__ = ["_render_interaction_map"]


def _render_interaction_map(
    im,
    apta_tokens,
    prot_tokens,
    apta_indices,
    prot_indices,
    ax=None,
    figsize=(20, 8),
):
    """Render a 2D interaction map heatmap with top-k token labels.

    Internal helper called by ``AptaTransPipeline.plot_interaction_map()``.
    Mirrors the plotting logic of the original AptaTrans authors' code.

    Parameters
    ----------
    im : numpy.ndarray
        2D interaction map of shape ``(n_apta_tokens, n_prot_tokens)``.
    apta_tokens : list[str]
        Decoded aptamer 3-mer token labels for the y-axis.
    prot_tokens : list[str]
        Decoded protein 3-mer token labels for the x-axis.
    apta_indices : list[int]
        Indices of top-k aptamer tokens to show as y-axis tick labels.
    prot_indices : list[int]
        Indices of top-k protein tokens to show as x-axis tick labels.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure and axes are created.
    figsize : tuple[int, int], optional, default=(20, 8)
        Figure size when a new figure is created (ignored if ``ax`` is given).

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the rendered heatmap.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.imshow(im, aspect="auto", cmap="viridis")
    ax.set_xticks(prot_indices)
    ax.set_xticklabels(
        [prot_tokens[i] for i in prot_indices], rotation=90, fontsize=8
    )
    ax.set_yticks(apta_indices)
    ax.set_yticklabels([apta_tokens[i] for i in apta_indices], fontsize=8)
    ax.set_xlabel("Protein")
    ax.set_ylabel("Aptamer")
    ax.figure.colorbar(ax.images[0], ax=ax)

    return ax
