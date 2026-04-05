"""Visualization utilities for the pyaptamer package."""

__all__ = ["plot_interaction_map"]

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def _get_protein_token_lengths(
    seq: str, words: dict[str, int | float], word_max_len: int = 3
) -> list[int]:
    """Helper function to reconstruct token lengths from greedy tokenizer."""
    token_lengths = []
    i = 0
    while i < len(seq):
        matched = False
        for pattern_len in range(min(word_max_len, len(seq) - i), 0, -1):
            if seq[i : i + pattern_len] in words:
                token_lengths.append(pattern_len)
                i += pattern_len
                matched = True
                break
        if not matched:
            token_lengths.append(1)
            i += 1
    return token_lengths


def plot_interaction_map(
    interaction_map: np.ndarray | torch.Tensor,
    candidate: str,
    target: str,
    prot_words: dict[str, int | float] | None = None,
    title: str = "Aptamer-Protein Interaction Map",
    cmap: str = "viridis",
    show: bool = True,
    **kwargs,
) -> plt.Figure:
    """Plot the interaction map between an aptamer candidate and a target protein.

    Parameters
    ----------
    interaction_map : np.ndarray | torch.Tensor
        The interaction map tensor/array. The raw output from AptaTransPipeline.
    candidate : str
        The candidate aptamer sequence.
    target : str
        The target protein sequence.
    prot_words : dict[str, int | float], optional, default=None
        The dictionary mapping protein n-mer subsequences to unique IDs. 
        Highly recommended to pass `pipeline.prot_words`. If provided, it is 
        used to accurately project the tokenizer's output back to single 
        amino-acid resolution as described in the AptaTrans paper.
    title : str, optional, default="Aptamer-Protein Interaction Map"
        The title of the plot.
    cmap : str, optional, default="viridis"
        The colormap for the heatmap.
    show : bool, optional, default=True
        Whether to show the plot using plt.show().
    **kwargs
        Additional keyword arguments passed to `seaborn.heatmap`.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure containing the heatmap.
    """
    if isinstance(interaction_map, torch.Tensor):
        imap = interaction_map.detach().cpu().numpy()
    else:
        imap = np.array(interaction_map)

    # Remove batch and channel dimensions if present
    imap = np.squeeze(imap)

    if imap.ndim != 2:
        raise ValueError(
            f"Expected a 2D interaction map after squeezing, got shape {imap.shape} "
            f"(original shape: {interaction_map.shape})."
        )

    # Calculate the number of unpadded tokens
    n_apta_tokens = max(1, len(candidate) - 2)
    
    if prot_words is not None:
        prot_token_lengths = _get_protein_token_lengths(target, prot_words)
        n_prot_tokens = len(prot_token_lengths)
    else:
        import warnings
        warnings.warn(
            "prot_words not provided. Reconstructing single amino-acid mapping may be inaccurate."
        )
        prot_token_lengths = [1] * len(target)
        n_prot_tokens = len(target)

    # Slice the valid (unpadded) region of the interaction map
    if imap.shape[0] >= n_apta_tokens and imap.shape[1] >= n_prot_tokens:
        imap_valid = imap[:n_apta_tokens, :n_prot_tokens]
    else:
        imap_valid = imap

    # AptaTrans Mapping 1: Aptamer nucleotide score is average of overlapping 3-mers
    reconstructed_apta = []
    dim_prot = imap_valid.shape[1]
    
    for j in range(len(candidate)):
        valid_i = [i for i in range(max(0, j - 2), min(n_apta_tokens, j + 1))]
        if valid_i:
            avg_score = np.mean(imap_valid[valid_i, :], axis=0)
        else:
            avg_score = np.zeros(dim_prot)
        reconstructed_apta.append(avg_score)
        
    imap_mapped = np.vstack(reconstructed_apta)

    # AptaTrans Mapping 2: Protein word score broadcasts to its containing amino acids
    if prot_words is not None:
        reconstructed_prot = []
        for col_idx, t_len in enumerate(prot_token_lengths):
            if col_idx < dim_prot:
                col = imap_mapped[:, col_idx : col_idx + 1]
                reconstructed_prot.append(np.tile(col, (1, t_len)))
        if reconstructed_prot:
            imap_mapped = np.hstack(reconstructed_prot)

    r_len = len(candidate)
    c_len = len(target)
    
    # Ensure dimensions match before plotting
    if imap_mapped.shape != (r_len, c_len):
        import warnings
        warnings.warn(f"Mapped imap shape {imap_mapped.shape} does not match sequences ({r_len}, {c_len}).")
        
    imap = imap_mapped

    # Initialize plot with a clean style
    with sns.axes_style("white"):
        fig_width = max(6.0, min(20.0, c_len * 0.4 + 2.0)) # Add some padding for the colorbar
        fig_height = max(5.0, min(16.0, r_len * 0.4 + 1.5))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot the heatmap with professional styling
        sns.heatmap(
            imap,
            cmap=cmap,
            square=True,              # Perfect square cells
            linewidths=0.5,           # Subtle grid lines
            linecolor="lightgray",
            xticklabels=list(target) if imap.shape[1] == c_len else "auto",
            yticklabels=list(candidate) if imap.shape[0] == r_len else "auto",
            cbar_kws={
                "label": "Interaction Intensity",
                "shrink": 0.8,        # Slightly smaller colorbar
                "aspect": 30          # Thinner colorbar
            },
            ax=ax,
            **kwargs,
        )

        # Better typography for labels and titles
        ax.set_xlabel("Protein residues", fontsize=12, fontweight="medium", labelpad=10)
        ax.set_ylabel("Aptamer nucleotides", fontsize=12, fontweight="medium", labelpad=10)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Improve layout and clean up ticks
        ax.tick_params(axis="x", rotation=0, labelsize=10, bottom=False)
        ax.tick_params(axis="y", rotation=0, labelsize=10, left=False)
        
        plt.tight_layout()

    if show:
        plt.show()

    return fig
