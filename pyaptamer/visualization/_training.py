__author__ = "Alleny244"
__all__ = ["plot_training_curves"]


def plot_training_curves(
    train_losses,
    val_losses=None,
    title="Training Curves",
    ax=None,
):
    """
    Plot training and optional validation loss curves.

    Parameters
    ----------
    train_losses : list of float
        Training loss values per epoch.
    val_losses : list of float, optional
        Validation loss values per epoch.
    title : str, default="Training Curves"
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted curves.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss")

    if val_losses is not None:
        ax.plot(epochs, val_losses, label="Val Loss")

    ax.set(
        xlabel="Epoch",
        ylabel="Loss",
        title=title,
    )
    ax.legend()

    return ax
