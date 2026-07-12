__author__ = "Alleny244"
__all__ = ["plot_benchmark_results"]

import numpy as np


def plot_benchmark_results(
    results,
    metric=None,
    title="Benchmark Results",
    ax=None,
):
    """
    Plot benchmark results as a grouped bar chart.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame returned by :meth:`Benchmarking.run` with a
        ``pd.MultiIndex`` of (estimator, metric) and columns
        ``["train", "test"]``.
    metric : str, optional
        If provided, filter results to a single metric. Otherwise all
        metrics are plotted.
    title : str, default="Benchmark Results"
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted benchmark results.
    """
    import matplotlib.pyplot as plt

    df = results.copy()
    if metric is not None:
        df = df.xs(metric, level="metric")
        x_labels = df.index.tolist()
    else:
        x_labels = [f"{est}\n{met}" for est, met in df.index]

    x = np.arange(len(x_labels))
    width = 0.35

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(x_labels) * 2), 5))

    ax.bar(x - width / 2, df["train"], width, label="Train")
    ax.bar(x + width / 2, df["test"], width, label="Test")

    ax.set(
        xticks=x,
        xticklabels=x_labels,
        ylabel="Score",
        title=title,
    )
    ax.legend()
    ax.set_ylim(0, 1.05)

    return ax
