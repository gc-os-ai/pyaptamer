__author__ = "Alleny244"
__all__ = ["plot_confusion_matrix", "plot_roc_curve"]

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    normalize=False,
    title="Confusion Matrix",
    cmap="Blues",
    ax=None,
):
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    labels : list of str, optional
        Display labels for the classes.
    normalize : bool, default=False
        If True, normalize the matrix by row (true class).
    title : str, default="Confusion Matrix"
        Title for the plot.
    cmap : str, default="Blues"
        Matplotlib colormap name.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted confusion matrix.
    """
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm.astype(float) / row_sums

    if ax is None:
        _, ax = plt.subplots()

    n_classes = cm.shape[0]
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    if labels is None:
        labels = [str(i) for i in range(n_classes)]

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    return ax


def plot_roc_curve(
    y_true,
    y_score,
    title="ROC Curve",
    ax=None,
):
    """
    Plot a Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_score : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    title : str, default="ROC Curve"
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted ROC curve.
    """
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.legend(loc="lower right")

    return ax
